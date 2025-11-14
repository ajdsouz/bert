from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
import numpy as np

from pathlib import Path


TOKEN_DTYPE = np.int32


def tokenize_dataset_as_memmap(
    memmap_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
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

    assert len(tokenizer) < np.iinfo(TOKEN_DTYPE).max
    assert len(dataset) < np.iinfo(TOKEN_DTYPE).max

    Path(memmap_path).parent.mkdir(parents=True, exist_ok=True)

    def tokenize_fn(text):
        # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
        tokens = tokenizer.encode(text)  # dataset.map will ignore tensor format
        return {"__tokens__": tokens, "__len__": len(tokens)}

    mapped_dataset = dataset.map(
        tokenize_fn,
        input_columns=input_columns,
        num_proc=num_tokenizing_proc,
        remove_columns=dataset.column_names,
    )

    seg = np.cumsum(
        [len(mapped_dataset) + 1] + list(mapped_dataset["__len__"]), dtype=TOKEN_DTYPE
    )
    memmap = np.memmap(memmap_path, dtype=TOKEN_DTYPE, mode="w+", shape=(seg[-1],))

    # Store the segments info at the head
    seg_beg = seg[0]
    memmap[:seg_beg] = seg

    for seg_end, tokens in zip(seg[1:], mapped_dataset["__tokens__"]):
        memmap[seg_beg:seg_end] = tokens
        seg_beg = seg_end

    memmap.flush()


class TokenizedMemmapDataset(torch.utils.data.Dataset):
    def __init__(self, memmap_path: str | Path) -> None:
        self._memmap = np.memmap(memmap_path, dtype=TOKEN_DTYPE, mode="r")
        self._segments = self._memmap[: self._memmap[0]]
        assert len(self._segments) > 1
        assert self._segments[-1] == len(self._memmap)

    def __len__(self) -> int:
        return len(self._segments) - 1

    def __getitem__(self, idx: int):
        seg_beg, seg_end = self._segments[idx : idx + 2]
        # torch.from_numpy will use the same memory from numpy, but if without .copy():
        # UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
        result = torch.from_numpy(self._memmap[seg_beg:seg_end].copy())
        return result
