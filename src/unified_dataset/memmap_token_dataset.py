from pathlib import Path
from typing import override

import numpy as np
import torch
import transformers
from typeguard import typechecked

from src.unified_dataset.interface import (
    _UnifiedDatasetConfigBase,
    UnifiedDatasetInterface,
)


class MemmapTokenDatasetConfig(_UnifiedDatasetConfigBase):
    block_size: int
    num_tokens: dict[str, int]
    tokenizer: str
    memmap_dtype: str


@typechecked
class MemmapTokenDataset(UnifiedDatasetInterface):
    conf: MemmapTokenDatasetConfig  # @property

    def __init__(
        self,
        dataset_config: MemmapTokenDatasetConfig,
        dataset_root: str | Path,
        split: str,
    ):
        super().__init__(dataset_config, dataset_root, split)
        assert self.conf.block_size > 2
        self._real_block_size = self.conf.block_size - 2

        path = self._root / self.conf.name / f"{self._split}.tokens"
        self._memmap = np.memmap(
            filename=path,
            dtype=getattr(np, self.conf.memmap_dtype),
            mode="r",
        )

        num_tokens = self.conf.num_tokens[split]
        if num_tokens == 0:
            num_tokens = len(self._memmap)
        assert 0 < num_tokens <= len(self._memmap)
        #assert num_tokens % self._real_block_size == 0
        self._num_tokens = num_tokens

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.conf.tokenizer)
        self._bos_token_id = int(tokenizer.bos_token_id)
        self._eos_token_id = int(tokenizer.eos_token_id)

    @override
    def __len__(self):
        return self._num_tokens // self._real_block_size

    @override
    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        """
        Returns:
            a dict contains
            - input_ids: `Int["block_size"]` tokens
        """
        beg = idx * self._real_block_size
        end = beg + self._real_block_size
        data = self._memmap[beg:end]
        tokens = np.empty(len(data) + 2, dtype=np.int64)
        tokens[0] = self._bos_token_id
        tokens[1:-1] = data
        tokens[-1] = self._eos_token_id
        return {"input_ids": torch.from_numpy(tokens)}
