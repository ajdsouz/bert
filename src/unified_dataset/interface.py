import abc
from pathlib import Path
from typing import Any, Iterable

import torch
from typeguard import typechecked

from src.utils import ConfigBase


class _UnifiedDatasetConfigBase(ConfigBase):
    name: str


_VALID_SPLITS = ("train", "validation", "test")
_registered_unified_dataset_classes = dict[str, type]()


@typechecked
class UnifiedDatasetInterface(torch.utils.data.Dataset, abc.ABC):
    def __init_subclass__(cls):
        super().__init_subclass__()
        if not cls.__name__.startswith("_"):
            _registered_unified_dataset_classes[cls.__name__] = cls

    def __init__(
        self,
        config: _UnifiedDatasetConfigBase,
        root: str | Path,
        split: str,
    ):
        super().__init__()
        if split not in _VALID_SPLITS:
            raise ValueError("unknown split", split, _VALID_SPLITS)

        self.__conf = config
        self._root = Path(root)
        self._split = split

        if not self._root.is_dir():
            raise FileNotFoundError(self._dataset_root)

    @property
    def conf(self):
        return self.__conf

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        pass


@typechecked
def load_unified_dataset(
    config: _UnifiedDatasetConfigBase,
    root: str | Path,
    splits: Iterable[str] = _VALID_SPLITS,
) -> dict[str, UnifiedDatasetInterface]:
    """Load datasets splits based on dataset configure

    Args:
        config: configure of dataset
        splits: splits to load.
            Choices: {"train", "validation", "test"}

    Returns:
        Datasets corresponding to `splits`.
    """
    assert config.type.endswith("Dataset")
    cls = typechecked(_registered_unified_dataset_classes[config.type])
    results = dict[str, UnifiedDatasetInterface]()
    for sp in splits:
        results[sp] = cls(config, root, sp)
    return results
