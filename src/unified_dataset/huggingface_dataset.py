from pathlib import Path
from typing import override

import datasets
from typeguard import typechecked

from src.unified_dataset.interface import (
    _UnifiedDatasetConfigBase,
    UnifiedDatasetInterface,
)


class HuggingfaceDatasetConfig(_UnifiedDatasetConfigBase):
    path: str
    subset_name: str | None
    input_columns: list[str]
    target_columns: list[str]
    splits: dict[str, str]


@typechecked
class HuggingfaceDataset(UnifiedDatasetInterface):
    conf: HuggingfaceDatasetConfig  # @property

    def __init__(
        self,
        dataset_config: HuggingfaceDatasetConfig,
        dataset_root: str | Path,
        split: str,
    ):
        super().__init__(dataset_config, dataset_root, split)
        self.dataset = datasets.load_dataset(
            self.conf.path,
            name=self.conf.subset_name,
            split=self.conf.splits[self._split],
        )
        self._input_columns = tuple(self.dataset[x] for x in self.conf.input_columns)
        self._target_columns = tuple(self.dataset[x] for x in self.conf.target_columns)

    @override
    def __len__(self):
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> dict[str, tuple]:
        return {
            "inputs": tuple(x[idx] for x in self._input_columns),
            "targets": tuple(x[idx] for x in self._target_columns),
        }
