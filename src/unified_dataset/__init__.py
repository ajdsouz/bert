from src.unified_dataset.interface import load_unified_dataset, UnifiedDatasetInterface

# trigger registration of subclasses
from src.unified_dataset import huggingface_dataset as _
from src.unified_dataset import memmap_token_dataset as _
