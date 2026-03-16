from typing import override

import torch
import transformers

from src.modeling import interface


# Base Model
class HuggingfaceModelConfig(interface._BaseModelConfigBase):
    hf_repo: str

    def to_hf_conf(self) -> transformers.PreTrainedConfig:
        hf_conf = transformers.AutoConfig.from_pretrained(self.hf_repo)
        assert getattr(hf_conf, "vocab_size", 0) > 0
        assert getattr(hf_conf, "max_position_embeddings", 0) > 0
        hf_conf.vocab_size = self.vocab_size
        hf_conf.max_position_embeddings = self.max_sequence_length
        return hf_conf


class HuggingfaceModel(interface.ModelInterface):
    @classmethod
    def _from_config(cls, hf_conf):
        return transformers.AutoModel.from_config(hf_conf)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


# downstream interface
class _HuggingfaceModelForDownstreamConfigBase(interface._ModelForDownstreamConfigBase):
    base_config: HuggingfaceModelConfig

    def to_hf_conf(self) -> transformers.PreTrainedConfig:
        hf_conf = self.base_config.to_hf_conf()
        dict_ = self.to_dict()
        for k in _HuggingfaceModelForDownstreamConfigBase.__dataclass_fields__.keys():
            dict_.pop(k)
        for k, v in dict_.items():
            setattr(hf_conf, k, v)
        return hf_conf


class _HuggingfaceModelForDownstreamBase(interface._ModelForDownstreamInterface):
    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)
        upstream: torch.nn.Module = HuggingfaceModelForMlm._from_config(config)
        base_model_type = type(HuggingfaceModel._from_config(config))
        base_module_name, base_module = None, None
        for name, module in upstream.named_modules():
            if isinstance(module, base_model_type):
                assert base_module is None, "should match only one"
                base_module_name, base_module = name, module
        assert base_module is not None, "should match one"
        self.__base_module_name = base_module_name
        setattr(self, self.__base_module_name, base_module)

    @override
    @property
    def _base_model(self) -> torch.nn.Module:
        return getattr(self, self.__base_module_name)

    @override
    @property
    def _base_model_hidden_size(self) -> int:
        return self.config.hidden_size


# masked language modeling
class HuggingfaceModelForMlmConfig(_HuggingfaceModelForDownstreamConfigBase):
    pass


class HuggingfaceModelForMlm(interface.ModelInterface):
    @classmethod
    def _from_config(cls, hf_conf):
        return transformers.AutoModelForMaskedLM.from_config(hf_conf)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return transformers.AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
