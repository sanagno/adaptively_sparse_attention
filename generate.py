import torch
from utils.utils import get_model
from transformers import GPT2Tokenizer
from train import TOKENIZER_ID, HUGGINGFACE_TOKEN
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpt2_type", type=str, default="gpt2")
    parser.add_argument("--sample", type=str, default="true")
    args = parser.parse_args()

    args.sample = args.sample.lower() == "true"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(
        TOKENIZER_ID, use_auth_token=HUGGINGFACE_TOKEN
    )

    model = get_model(
        args.model_path,
        tokenizer.vocab_size,
        entmax_set_inf=True,
        device=device,
        checkpoint=args.checkpoint,
        gpt2_type=args.gpt2_type,
        verbose=True,
    )
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token

    # Get text through input
    while True:
        texts = [input("Enter text: ")]

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

        input_ids = inputs.input_ids
        mask = inputs.attention_mask

        generated_text, times_default, _, _ = model.generate(
            input_ids.to(device),
            max_length=1024,
            do_sample=args.sample,
            validity_mask=mask,
            measure_time=True,
        )

        print(
            "Generated text:",
            tokenizer.decode(generated_text[0].cpu().numpy(), skip_special_tokens=True),
        )


if __name__ == "__main__":
    main()
