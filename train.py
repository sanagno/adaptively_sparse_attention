import torch
import os
import numpy as np
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
from model import GPT, GPTConfig
from utils.entmax_scheduler import get_entmax_weight_scheduler
import argparse
import json
from tqdm import tqdm
import gc

HUGGINGFACE_TOKEN = "..."
TOKENIZER_ID = "..."
DATASET_ID = "..."

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


def collate_fn(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch])
    targets = torch.roll(input_ids, -1, dims=-1)
    # here we ignore masking as it is already taken care of in the dataset
    targets[:, -1] = -1  # ignore index is set to -1
    return input_ids, targets


def get_total_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.detach().cpu().norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def get_dataset(args):
    test_dataset = load_dataset(
        DATASET_ID,
        split=f"train[0:{args.test_size}]",
        cache_dir=args.cache_dir,
    )
    train_dataset = load_dataset(
        DATASET_ID,
        split=f"train[{args.test_size}:]",
        cache_dir=args.cache_dir,
    )

    return train_dataset, test_dataset


def get_dataloaders(args, train_dataset, test_dataset):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


@torch.no_grad()
def eval_model(model, test_dataloader, max_iters=None):
    model.eval()
    losses = []
    accs = []
    for cnt, test_inputs in enumerate(test_dataloader):
        input_ids, targets = test_inputs[0].cuda(), test_inputs[1].cuda()

        logits, _, loss, _ = model(input_ids, targets=targets)

        losses.append(loss.item())

        preds = torch.argmax(logits, axis=-1)
        acc = (preds == targets).float().mean().item()

        accs.append(acc)

        del loss, logits, preds, acc, input_ids, targets

        if max_iters is not None and cnt >= max_iters:
            break

    return np.mean(losses), np.mean(accs)


def get_sparsity_loss(all_cumprobs, sparse_attention, sparsity_loss_weight):
    if not sparse_attention:
        return 0

    # Ignore the last token as we do not make any predictions from it due to the shift
    all_cumprobs = torch.stack(all_cumprobs, dim=0)

    return (all_cumprobs[:, :, :, :-1, :] * sparsity_loss_weight).mean()


def main(args):
    torch.manual_seed(args.seed)

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise ValueError("No GPU available")

    tokenizer = GPT2Tokenizer.from_pretrained(
        TOKENIZER_ID, use_auth_token=HUGGINGFACE_TOKEN
    )

    if args.pretrained:
        model = GPT.from_pretrained(
            args.pretrained,
            override_args={
                "sparse_attention": args.sparse_attention,
                "sparse_attention_int_bias": args.sparse_attention_int_bias,
                "int_n_embd": args.int_n_embd,
                "propagate_in_depth": args.propagate_in_depth,
            },
            cache_dir=args.cache_dir,
        )
    else:
        configuration = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            sparse_attention=args.sparse_attention,
            n_layer=args.n_layer,
            sparse_attention_int_bias=args.sparse_attention_int_bias,
            int_n_embd=args.int_n_embd,
            propagate_in_depth=args.propagate_in_depth,
        )

        model = GPT(configuration)

    print(model)

    ## Get loading directory
    cnt = 0
    log_dir = args.log_dir + f"/runs_{cnt}/"

    while os.path.exists(log_dir):
        cnt += 1
        log_dir = args.log_dir + f"/runs_{cnt}/"

    os.makedirs(log_dir)

    argparse_dict = vars(args).copy()
    with open(os.path.join(log_dir, "config.json"), "w") as fp:
        json.dump(argparse_dict, fp, ensure_ascii=False, indent=4)

    writer = SummaryWriter(log_dir)
    writer.add_text("args", str(args))

    ## Get data
    train_dataset, test_dataset = get_dataset(args)

    train_dataloader, test_dataloader = get_dataloaders(
        args, train_dataset, test_dataset
    )

    if args.optimizer == "adam":
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: min(
                1.0, step / args.warmup_steps
            ),  # Linear warmup over warmup_steps.
        )
    else:
        scheduler = None

    entmax_weight_scheduler = (
        get_entmax_weight_scheduler(args.entmax_weight_scheduler)
        if args.entmax_weight_scheduler
        else None
    )

    global_step = 0

    pbar = tqdm(total=args.max_steps)

    iterator = iter(train_dataloader)

    model = model.cuda()

    device = torch.cuda.current_device()

    while global_step <= args.max_steps:
        ### Update alphas in the alpha-sigmoid
        if args.entmax_weight_scheduler:
            val = entmax_weight_scheduler(global_step)

            for layer_i in range(len(model.transformer.h)):
                model.transformer.h[layer_i].attn.sparsity_alpha = val

        for step in range(args.micro_steps):
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator.close()
                # force garbage collection
                del iterator
                gc.collect()

                iterator = iter(train_dataloader)
                inputs = next(iterator)

            input_ids, targets = inputs[0].to(device), inputs[1].to(device)

            # with ctx:
            logits, all_cumprobs, loss, _ = model(
                input_ids,
                targets=targets,
            )

            sparisty_loss = get_sparsity_loss(
                all_cumprobs,
                args.sparse_attention,
                args.sparsity_loss_weight,
            )

            total_loss = loss + sparisty_loss

            total_loss = total_loss / args.micro_steps

            total_loss.backward()

            del (
                total_loss,
                all_cumprobs,
                input_ids,
            )

            if step != args.micro_steps - 1:
                del targets, logits, loss, sparisty_loss

        if global_step % args.logging_step == 0:
            print(f"Step {global_step} loss {loss} sparisty_loss {sparisty_loss}")

        if global_step % args.logging_step == 0 and not args.debug:
            writer.add_scalar("loss/train", loss.float(), global_step)
            writer.add_scalar(
                "loss/sparisty_loss",
                sparisty_loss.float()
                if torch.is_tensor(sparisty_loss)
                else sparisty_loss,
                global_step,
            )

            if args.entmax_weight_scheduler:
                writer.add_scalar("entmax_weight", val, global_step)

            preds = torch.argmax(logits, axis=-1)
            acc = (preds == targets).float().mean().item()
            writer.add_scalar("acc/train", acc, global_step)

        if global_step % args.logging_step == 0:
            writer.add_scalar(
                "grad_norm_before", get_total_grad_norm(model), global_step
            )

        if args.clip is not None and args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if global_step % args.logging_step == 0:
            writer.add_scalar(
                "grad_norm_after", get_total_grad_norm(model), global_step
            )

        opt.step()
        opt.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if global_step % args.save_every == 0 and global_step > 0 and not args.debug:
            torch.save(
                model.state_dict(),
                os.path.join(log_dir, f"model_step_{global_step}.pt"),
            )

        if global_step % args.eval_every == 0 and not args.debug:
            loss, acc = eval_model(model, test_dataloader, max_iters=10)
            writer.add_scalar("loss/test", loss, global_step)
            writer.add_scalar("acc", acc, global_step)

            model.train()

        del loss, sparisty_loss, logits, targets

        if global_step > args.max_steps:
            break

        global_step += 1
        pbar.update(1)

        # log learning rate
        writer.add_scalar("lr", opt.param_groups[0]["lr"], global_step)

    torch.save(
        model.state_dict(),
        os.path.join(log_dir, f"last_model_step_{global_step}.pt"),
    )


def get_arguments(notebook=False, notebook_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="test")
    parser.add_argument("--base_log_dir", type=str, default="logs")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=25000)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2000)

    parser.add_argument("--save_every", type=int, default=25000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--entmax_weight_scheduler", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--sparse_attention", type=str, default="false")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--sparse_attention_int_bias", type=float, default=2.0)
    parser.add_argument("--sparsity_loss_weight", type=float, default=0.3)

    parser.add_argument("--micro_steps", type=int, default=1)
    parser.add_argument("--int_n_embd", type=int, default=None)

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--propagate_in_depth", type=str, default="false")

    if notebook:
        args = parser.parse_known_args(notebook_args)[0]
    else:
        args = parser.parse_args()

    args.log_dir = os.path.join(
        args.base_log_dir, args.log_dir if not args.debug else "debug"
    )
    args.sparse_attention = args.sparse_attention.lower() == "true"
    args.propagate_in_depth = args.propagate_in_depth.lower() == "true"
    args.dataset_id = DATASET_ID  # for logging purposes

    return args


if __name__ == "__main__":
    args = get_arguments()
    main(args)
