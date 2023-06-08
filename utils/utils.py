import os
import torch
from argparse import Namespace
import json
from model import GPT, GPTConfig
from .entmax_scheduler import get_entmax_weight_scheduler

config_args = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}


def get_model(
    base_dir,
    vocab_size,
    entmax_set_inf=True,
    device="cuda:0",
    checkpoint=None,
    gpt2_type="gpt2",
    verbose=False,
):
    if checkpoint is None:
        # List all checkpoints
        checkpoint = list(
            filter(lambda x: x.startswith("last_model_step"), os.listdir(base_dir))
        )[0]
        global_step = int(checkpoint.split("_")[-1].split(".")[0])

        print(f"Loding checkpoint {checkpoint}")

        checkpoint = os.path.join(base_dir, checkpoint)
    else:
        checkpoint = os.path.join(base_dir, checkpoint)

    with open(os.path.join(base_dir, "config.json"), "r") as f:
        args = Namespace(**json.load(f))

    if verbose:
        print("Loaded args", args)

    configuration = GPTConfig(
        vocab_size=vocab_size,
        sparse_attention=args.sparse_attention,
        int_n_embd=args.int_n_embd,
    )

    # update with config_args
    configuration.n_layer = config_args[gpt2_type]["n_layer"]
    configuration.n_head = config_args[gpt2_type]["n_head"]
    configuration.n_embd = config_args[gpt2_type]["n_embd"]

    model = GPT(configuration)

    print(f"Loading checkpoint {checkpoint}")
    print(
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    )

    if args.entmax_weight_scheduler:
        if entmax_set_inf:
            # we could set val to something really big, but it is unstable, also slower ...
            val = "inf"
        else:
            entmax_weight_scheduler = get_entmax_weight_scheduler(
                args.entmax_weight_scheduler
            )

            val = entmax_weight_scheduler(global_step)

        print("Setting sparse-sigmoid alpha to value", val)

        for layer_i in range(len(model.transformer.h)):
            model.transformer.h[layer_i].attn.sparsity_alpha = val
    else:
        print("No sparse sigmoid scheduler")

    model.to(device)

    return model
