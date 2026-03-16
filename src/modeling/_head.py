from typing import override

import torch
import transformers

from src.modeling._annotation import T, FP, Int, tensor_typechecked, typechecked
from src.modeling.interface import (
    _ModelForDownstreamInterface,
    _ModelForDownstreamConfigBase,
)


# sequence classification
class _SequenceClassificationConfig(_ModelForDownstreamConfigBase):
    num_labels: int


@typechecked
class _SequenceClassificationHead(_ModelForDownstreamInterface):
    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)
        self.sequence_classifier = torch.nn.Linear(
            self._base_model_hidden_size, self.config.num_labels
        )
        self.post_init()

    @tensor_typechecked
    @override
    def forward(
        self,
        *,
        input_ids: Int[T, "batch seq"],
        attention_mask: Int[T, "batch seq"],
        labels: Int[T, "batch"] | None = None,
    ) -> transformers.modeling_outputs.SequenceClassifierOutput:
        # position 0 must be <bos> token
        base_output: transformers.modeling_outputs.BaseModelOutput = self._base_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        x: FP[T, "batch seq hidden"] = base_output.last_hidden_state
        logits: FP[T, "batch num_labels"] = self.sequence_classifier(x[:, 0, :])

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels
            )

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_output.hidden_states,
            attentions=base_output.attentions,
        )


# token classification
class _TokenClassificationConfig(_ModelForDownstreamConfigBase):
    num_labels: int


@typechecked
class _TokenClassificationHead(_ModelForDownstreamInterface):
    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)
        self.token_classifier = torch.nn.Linear(
            self._base_model_hidden_size, self.config.num_labels
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
    ) -> transformers.modeling_outputs.TokenClassifierOutput:
        base_output: transformers.modeling_outputs.BaseModelOutput = self._base_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        x: FP[T, "batch seq hidden"] = base_output.last_hidden_state
        logits: FP[T, "batch seq num_labels"] = self.token_classifier(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )

        return transformers.modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_output.hidden_states,
            attentions=base_output.attentions,
        )


# question answering
class _QuestionAnsweringConfig(_ModelForDownstreamConfigBase):
    pass


@typechecked
class _QuestionAnsweringHead(_ModelForDownstreamInterface):
    def __init__(self, config: transformers.PreTrainedConfig):
        super().__init__(config)
        self.qa_head = torch.nn.Linear(self._base_model_hidden_size, 2)
        self.post_init()

    @tensor_typechecked
    @override
    def forward(
        self,
        *,
        input_ids: Int[T, "batch seq"],
        attention_mask: Int[T, "batch seq"],
        start_positions: Int[T, "batch"] | None = None,
        end_positions: Int[T, "batch"] | None = None,
    ) -> transformers.modeling_outputs.QuestionAnsweringModelOutput:
        assert not ((start_positions is None) ^ (end_positions is None))
        base_output: transformers.modeling_outputs.BaseModelOutput = self._base_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        x: FP[T, "batch seq hidden"] = base_output.last_hidden_state
        logits: FP[T, "batch seq 2"] = self.qa_head(x)
        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        loss = None
        if start_positions is not None:
            start_loss = torch.nn.functional.cross_entropy(
                start_logits, start_positions
            )
            end_loss = torch.nn.functional.cross_entropy(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return transformers.modeling_outputs.QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=base_output.hidden_states,
            attentions=base_output.attentions,
        )
