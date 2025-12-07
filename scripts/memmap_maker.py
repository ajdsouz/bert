import argparse
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
# import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output_memmap_path", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--subset", type=str, default=None)
# parser.add_argument("--dataset_split", "-s", type=str, default="train[:10]")
parser.add_argument(
    "--dataset_columns", "-c", type=str, default=["text"]
)
parser.add_argument("--val_ratio", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--tokenizer", default="FacebookAI/roberta-base")
args = parser.parse_args()

def batch_iter(dataset, batch_size, columns):
    batch = []
    for ex in dataset:
        text = " ".join(str(ex[c]) for c in columns)
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_tokens(dataset, tokenizer, batch_size, columns):
    total = 0
    for batch in tqdm(batch_iter(dataset, batch_size, columns), desc="Counting tokens"):
        enc = tokenizer(batch, add_special_tokens=False)
        for ids in enc["input_ids"]:
            total += len(ids)
    return total


def write_tokens(dataset, tokenizer, mmap, batch_size, columns):
    idx = 0
    for batch in tqdm(batch_iter(dataset, batch_size, columns), desc="Writing memmap"):
        enc = tokenizer(batch, add_special_tokens=False)
        for ids in enc["input_ids"]:
            arr = np.array(ids, dtype=np.uint32)
            mmap[idx: idx + len(arr)] = arr
            idx += len(arr)
    mmap.flush()

def memmap_dataset(
    memmap_file_path: str | Path,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    input_columns: str | list[str],
    batch_size: int = 8
) -> None:
    """
    Tokenize the dataset and store it as memmap.
    - The result sturcture of memmap file, in an 1D np.ndarray with TOKEN_DTYPE:
        - 0:        length of the segments table, aka. offset to the begin of first sample
        - 1...$[0]: Offsets to the begin of the sample with corresponding index
        - $[0]...:  concatenated tokenized samples
    - Args:
        - memmap_path: Path to the result memmap file.
        - tokenizer: HuggingFace tokenizer object.
        - dataset: HuggingFace dataset object.
        - input_columns: Target columns of the dataset to be tokenized.
        - num_tokenizing_proc: Number of process to tokenize (HuggingFace built-in)
    """

    input_columns = args.dataset_columns
    if isinstance(input_columns, str):
        input_columns = [input_columns]
        
    for split, data in dataset.items():
        
        print(f"---Processing {split}---")
        token_count = count_tokens(data, tokenizer, batch_size, input_columns)

        filename = f"{memmap_file_path}/{split}.tokens"
        memmap_file = np.memmap(filename=filename, dtype=np.uint32, mode='w+', shape=(token_count,))

        write_tokens(data, tokenizer, memmap_file, batch_size, input_columns)
        

def make_memmap_dataset(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # dataset = load_dataset(*args.dataset_args, split=args.dataset_split)
    subset = None if args.subset=="None" else args.subset
    print(f"using {args.tokenizer} on {args.dataset} subset {args.subset} column {args.dataset_columns}")
    dataset = load_dataset(args.dataset, subset)

    if len(dataset.keys())<2 and 'train' in dataset.keys():
        split_dataset = dataset['train'].train_test_split(test_size=args.val_ratio, seed=1337)
        split_dataset['validation'] = split_dataset.pop('test')
        memmap_dataset(
            args.output_memmap_path, tokenizer, split_dataset, args.dataset_columns, args.batch_size
        )
    else:
        memmap_dataset(
        args.output_memmap_path, tokenizer, dataset, args.dataset_columns, args.batch_size
        )

make_memmap_dataset(args=args)
