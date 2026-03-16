from typing import override

import torch
import transformers
from transformers.modeling_outputs import BaseModelOutput

from src.modeling._annotation import T, FP, Int, tensor_typechecked, typechecked
from src.modeling import interface
from src.modeling.my_bert import _layers


# base model
class MyBertModelConfig(interface._BaseModelConfigBase):
    hidden_size: int
    num_encoder_layers: int
    num_attention_heads: int
    gated_attention: bool
    mlp_type: str


@typechecked
class MyBertModel(interface.ModelInterface):
    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)
        self.token_embedding = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.positional_encoding = _layers.SinusoidalPositionalEncoding(
            self.config.max_sequence_length, self.config.hidden_size
        )
        self.layernorm_embedding = torch.nn.LayerNorm(self.config.hidden_size)
        self.encoder_layers = torch.nn.ModuleList()
        self.layernorm_last = torch.nn.LayerNorm(self.config.hidden_size)
        for _ in range(self.config.num_encoder_layers):
            self.encoder_layers.append(_layers.EncoderLayer(
                self.config.hidden_size,
                self.config.num_attention_heads,
                self.config.gated_attention,
                self.config.mlp_type,
            ))  # fmt:skip
        self.post_init()

    @tensor_typechecked
    @override
    def forward(
        self,
        *,
        input_ids: Int[T, "batch seq"],
        attention_mask: Int[T, "batch seq"],
    ) -> BaseModelOutput:
        # Check transformers.PreTrainedModel._init_weights
        # Note that the weight of token embedding is normalized with std=0.02 by default
        # But the range of positional_encoding is [-0.5~0.5]
        # So we need to time it by hidden_size**0.5
        x: FP[T, "batch seq hidden"] = self.token_embedding(input_ids)
        x = self.positional_encoding(x * self.config.hidden_size**0.5)
        x = self.layernorm_embedding(x)

        hidden_states: list[FP[T, "batch seq hidden"]] = [x]
        attentions = list[FP[T, "batch seq hidden"]]()
        for layer in self.encoder_layers:
            x, attention_value = layer(x, attention_mask)
            hidden_states.append(x)
            attentions.append(attention_value)

        last_hidden_state = self.layernorm_last(hidden_states[-1])
        if not self.config.output_hidden_states:
            hidden_states = None
        if not self.config.output_attentions:
            attentions = None

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states and tuple(hidden_states),
            attentions=attentions and tuple(attentions),
        )


# downstream interfaces
class _MyBertModelForDownstreamConfigBase(interface._ModelForDownstreamConfigBase):
    base_config: MyBertModelConfig


class _MyBertModelForDownstreamBase(interface._ModelForDownstreamInterface):
    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)
        self.base = MyBertModel(config.base_config)
        self.post_init()

    @override
    @property
    def _base_model(self):
        return self.base

    @override
    @property
    def _base_model_hidden_size(self) -> int:
        return self.config.base_config.hidden_size


# masked language modeling
class MyBertModelForMlmConfig(_MyBertModelForDownstreamConfigBase):
    tie_word_embeddings: bool


@typechecked
class MyBertModelForMlm(_MyBertModelForDownstreamBase):
    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)
        self.transform_layer = torch.nn.Identity()
        self.mlm_head = torch.nn.Linear(
            self.hidden_size, self.config.base_config.vocab_size
        )
        self._tied_weights_keys = dict[str, str]()
        if self.config.tie_word_embeddings:
            self._tied_weights_keys = {"mlm_head.weight": "base.token_embedding.weight"}
            self.transform_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.GELU(),
                torch.nn.LayerNorm(self.hidden_size),
            )
        self.post_init()

    @tensor_typechecked
    @override
    def forward(
        self,
        *,
        input_ids: Int[T, "batch seq"],
        attention_mask: Int[T, "batch seq"],
        labels: Int[T, "batch seq"] | None = None,
    ) -> transformers.modeling_outputs.MaskedLMOutput:
        base_output: transformers.modeling_outputs.BaseModelOutput = self.base(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden: FP[T, "batch seq hidden"] = self.transform_layer(
            base_output.last_hidden_state
        )
        logits: FP[T, "batch seq vocab"] = self.mlm_head(hidden)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )

        return transformers.modeling_outputs.MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_output.hidden_states,
            attentions=base_output.attentions,
        )
