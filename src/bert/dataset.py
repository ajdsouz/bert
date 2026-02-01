import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self, memmap_path: str, block_size: int, num_tokens: int | None) -> None:
        super().__init__()
        data = np.memmap(filename=memmap_path, dtype=np.uint16, mode='r')
        if num_tokens is not None:
            data = data[:num_tokens]
        self.data: np.memmap = data
        self.block_size: int = block_size
        self.num_sequences: int = (len(self.data) -1) // block_size
        self.start_indices: np.ndarray = np.arange(self.num_sequences) * block_size

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        start = self.start_indices[index]
        tokens: Tensor = torch.from_numpy(self.data[start:(start + self.block_size)].copy()).long() # TODO : should we use .copy()? 
        return {'input_ids':tokens}