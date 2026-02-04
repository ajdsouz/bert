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
    
    
class TokenDatasetV2(Dataset):
    def __init__(self, memmap_path: str, block_size: int, num_tokens: int | None, bos_token_id: int, eos_token_id: int) -> None:
        super().__init__()
        data = np.memmap(filename=memmap_path, dtype=np.uint16, mode='r')
        if num_tokens is not None:
            data = data[:num_tokens]
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data: np.memmap = data
        self.block_size: int = block_size -2
        self.num_sequences: int = (len(self.data) -1) // self.block_size
        self.start_indices: np.ndarray = np.arange(self.num_sequences) * self.block_size

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index: int) -> dict:
        """
        A PyTorch dataset for loading tokens from np.memmap files
        Modified to have a <s> at the start and </s> at the end of each sequence

        BLOCK SIZE DOESNT NEED TO BE REDUCED

        Args:
            index (_type_): _description_

        Returns:
            dict: _description_
        """
        start = self.start_indices[index]
        token_ids: np.ndarray = self.data[start:(start + self.block_size)].copy()
        sequence = np.empty(self.block_size+2, dtype=token_ids.dtype)
        sequence[0] = self.bos_token_id
        sequence[1:-1] = token_ids
        sequence[-1] = self.eos_token_id
        tokens: Tensor = torch.from_numpy(sequence).long()
        return {'input_ids':tokens}