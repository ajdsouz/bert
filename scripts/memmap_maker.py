import argparse
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
# import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


TOKEN_DTYPE = np.uint16
BATCH_SIZE = 1024

def memmap_dataset(
    memmap_file_path: str | Path,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    input_columns: str | list[str],
    num_tokenizing_proc: int = 0,
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

    def process(batch)-> dict:
        tokens = tokenizer(batch['text'], padding=False, truncation=False)
        input_ids = tokens['input_ids']
        return {
            'token_ids': input_ids,
            'len': [len(t) for t in input_ids]
        }
    
    dataset = dataset.map(
        process,
        batched=True,
        remove_columns=['text']
    )

    for split, data in dataset.items():
        tensor_length = tensor_length = sum(len(x) for x in data['token_ids'])
        filename = f"{memmap_file_path}/{split}.tokens"
        memmap_file = np.memmap(filename=filename, dtype=np.uint16, mode='w+', shape=(tensor_length,))
        idx = 0
        for batch_idx in tqdm(range(BATCH_SIZE), desc=f"Writing {filename}"):
            batch = data.shard(BATCH_SIZE, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['token_ids'])
            memmap_file[idx:idx+len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        memmap_file.flush()


parser = argparse.ArgumentParser()
parser.add_argument("--output_memmap_path", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--subset", type=str)
# parser.add_argument("--dataset_split", "-s", type=str, default="train[:10]")
parser.add_argument(
    "--dataset_columns", "-c", type=str, default=["text"]
)
parser.add_argument("--tokenizer", default="FacebookAI/roberta-base")
args = parser.parse_args()

def make_memmap_dataset(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # dataset = load_dataset(*args.dataset_args, split=args.dataset_split)

    print(f"using {args.tokenizer} on {args.dataset} subset {args.subset} column {args.dataset_columns}")
    dataset = load_dataset(args.dataset, args.subset)
    memmap_dataset(
        args.output_memmap_path, tokenizer, dataset, args.dataset_columns
    )

make_memmap_dataset(args=args)
