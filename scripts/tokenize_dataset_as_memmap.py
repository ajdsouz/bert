from transformers import AutoTokenizer
from datasets import load_dataset

import argparse

from src.dataset.tokenized_memmap import tokenize_dataset_as_memmap


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = load_dataset(*args.dataset_args, split=args.dataset_split)
    tokenize_dataset_as_memmap(
        args.output_memmap_path, tokenizer, dataset, args.dataset_columns
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_memmap_path", type=str)
    parser.add_argument("dataset_args", type=str, nargs="+")
    parser.add_argument("--dataset_split", "-s", type=str, default="train[:10]")
    parser.add_argument(
        "--dataset_columns", "-c", type=str, nargs="+", default=["text"]
    )
    parser.add_argument("--tokenizer_path", default="openai-community/gpt2")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
