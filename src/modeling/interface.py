import abc
from pathlib import Path

import torch
import transformers

from src.utils import ConfigBase
from src.modeling._annotation import T, Int, typechecked


class _ModelConfigBase(ConfigBase):
    def to_hf_conf(self) -> transformers.PreTrainedConfig:
        result = transformers.PreTrainedConfig()
        dict_ = self.to_dict()
        if dict_.pop("base_config", None) is not None:
            result.base_config = self.base_config.to_hf_conf()
        result.model_type = dict_.pop("type")
        for k, v in dict_.items():
            setattr(result, k, v)
        return result


# upstream model interface
class _BaseModelConfigBase(_ModelConfigBase):
    device: str
    dtype: str
    vocab_size: int
    max_sequence_length: int
    output_hidden_states: bool
    output_attentions: bool


# model_name => model_task => model_class
_registered_model_classes = dict[str, dict[str, type]]()


@typechecked
class ModelInterface(transformers.PreTrainedModel, abc.ABC):
    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__.startswith("_"):
            return
        if cls.__name__.endswith("Model"):
            _registered_model_classes[cls.__name__] = {"": cls}
            return

        idx = cls.__name__.find("ModelFor")
        assert idx != -1
        model_name = cls.__name__[: idx + 5]
        model_task = cls.__name__[idx + 8 :]
        assert model_name in _registered_model_classes
        _registered_model_classes[model_name][model_task] = cls

    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)

    @abc.abstractmethod
    def forward(
        self,
        *,
        input_ids: Int[T, "batch seq"],
        attention_mask: Int[T, "batch seq"],
    ) -> transformers.utils.ModelOutput:
        pass


# downstream model interface
class _ModelForDownstreamConfigBase(_ModelConfigBase):
    base_config: _BaseModelConfigBase


class _ModelForDownstreamInterface(ModelInterface):
    @property
    @abc.abstractmethod
    def _base_model(self) -> ModelInterface | transformers.PreTrainedModel:
        pass

    @property
    @abc.abstractmethod
    def _base_model_hidden_size(self) -> int:
        pass


@typechecked
def load_model(
    config: _ModelConfigBase,
    pretrained_path: str | Path | None = None,
) -> ModelInterface | transformers.PreTrainedModel:
    """Build model based on model configure"""
    hf_conf = config.to_hf_conf()

    device = torch.device(getattr(config, "base_config", config).device)
    dtype = getattr(torch, getattr(config, "base_config", config).dtype)
    assert isinstance(dtype, torch.dtype)

    config_type = config.type
    if config_type.endswith("Model"):
        cls = _registered_model_classes[config_type][""]
        assert issubclass(cls, transformers.PreTrainedModel)
        if pretrained_path is None:
            model = cls._from_config(hf_conf)
        else:
            model, info = cls.from_pretrained(
                pretrained_path, config=hf_conf, output_loading_info=True
            )
            for v in info.values():
                assert not v
        return model.to(device=device, dtype=dtype)

    idx = config_type.find("ModelFor")
    assert idx != -1
    model_name = config_type[: idx + 5]
    model_task = config_type[idx + 8 :]
    assert model_name in _registered_model_classes
    cls = _registered_model_classes[model_name][model_task]
    assert issubclass(cls, transformers.PreTrainedModel)

    if pretrained_path is None:
        model = cls._from_config(hf_conf)
    else:
        model, info = cls.from_pretrained(
            pretrained_path, config=hf_conf, output_loading_info=True
        )
        assert info.keys() == {
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        }
        assert not info["mismatched_keys"]
        assert not info["error_msgs"]
        # only train new modules
        for n, p in model.named_parameters():
            if n not in info["missing_keys"]:
                p.requires_grad_(False)
    return model.to(device=device, dtype=dtype)


@typechecked
def counting_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Return:
        A tuple contains
        - number of trainable parameters
        - number of all parameters
    """

    # https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L833
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param
