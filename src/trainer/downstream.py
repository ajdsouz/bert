from typing import Any, Callable, override

import evaluate
import transformers
from typeguard import typechecked

from src.unified_dataset import UnifiedDatasetInterface
from src.trainer.base import _TrainerConfigBase, TrainerBase


class _TrainerWithDatasetPreprocessor(TrainerBase):
    class PreprocessedDataset:
        def __init__(
            self,
            dataset: UnifiedDatasetInterface,
            data_processor: Callable[[dict[str, Any]], dict[str, Any]],
        ):
            self._items = list[dict]()
            num_discarded = 0
            for data in dataset:
                assert data.keys() == {"inputs", "targets"}
                processed = data_processor(data)
                if processed is None:
                    num_discarded += 1
                    continue
                assert {"input_ids"}.issubset(processed.keys())
                self._items.append(processed)
            if num_discarded > 0:
                print(
                    f"[WARNING] {num_discarded} of {len(dataset)} items in the dataset are discarded."
                )

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx: int):
            return self._items[idx]

    def _make_dataset_wrap(
        self, dataset: UnifiedDatasetInterface
    ) -> PreprocessedDataset:
        return self.PreprocessedDataset(dataset, self._preprocess_data)

    def _preprocess_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError


# masked language modeling
class MlmTrainerConfig(_TrainerConfigBase):
    mlm_probability: float


@typechecked
class MlmTrainer(TrainerBase):
    @override
    def _make_data_collator(self):
        return transformers.DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=True,
            mlm_probability=self.conf.mlm_probability,
            seed=self.conf.seed,
        )


# sequence classification
class SequenceClassificationTrainerConfig(_TrainerConfigBase):
    pass


@typechecked
class SequenceClassificationTrainer(_TrainerWithDatasetPreprocessor):
    @override
    def _preprocess_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        inputs, targets = data["inputs"], data["targets"]
        assert len(inputs) > 0 and len(targets) == 1
        assert isinstance(inputs[0], str)
        assert isinstance(targets[0], int)
        input_ids: list[int] = [self._tokenizer.bos_token_id]
        for inp in inputs:
            input_ids += self._tokenizer.encode(inp, add_special_tokens=False)
            input_ids.append(self._tokenizer.sep_token_id)
        input_ids[-1] = self._tokenizer.eos_token_id
        if len(input_ids) > self._tokenizer.model_max_length:
            return None  # discard long sentences
        return {"input_ids": input_ids, "labels": targets[0]}


# token classification
class TokenClassificationTrainerConfig(_TrainerConfigBase):
    pass


@typechecked
class TokenClassificationTrainer(_TrainerWithDatasetPreprocessor):
    @override
    def _preprocess_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        inputs, targets = data["inputs"], data["targets"]
        assert len(inputs) == len(targets) == 1
        assert isinstance(inputs[0], list) and isinstance(inputs[0][0], str)
        assert isinstance(targets[0], list) and isinstance(targets[0][0], int)

        out = self._tokenizer(
            data["inputs"][0],
            add_special_tokens=False,
            truncation=False,
            is_split_into_words=True,
        )
        input_ids: list[int] = out["input_ids"]
        assert isinstance(input_ids, list) and isinstance(input_ids[0], int)
        # assert out["attention_mask"] == [1] * len(input_ids)  # Useless without padding

        orig_upos: list[int] = data["targets"][0]
        labels: list[int] = [-100] * len(input_ids)
        prev_word_ids = None
        for idx, word_ids in enumerate(out.word_ids()):
            # Special tokens have a word id that is None. We set the label to pad_label_id
            if word_ids is None:
                continue
            # We set the label for the first token of each word.
            if prev_word_ids != word_ids:
                prev_word_ids = word_ids
                labels[idx] = orig_upos[word_ids]
            # For the other tokens in a word, we also set the label to pad_label_id

        input_ids = [
            self._tokenizer.bos_token_id,
            *input_ids,
            self._tokenizer.eos_token_id,
        ]
        labels = [-100, *labels, -100]
        if len(input_ids) > self._tokenizer.model_max_length:
            return None  # discard long sentences
        return {"input_ids": input_ids, "labels": labels}

    def _make_data_collator(self) -> Callable[[list[dict[str, Any]]], dict] | None:
        return transformers.DataCollatorForTokenClassification(self._tokenizer)


# question answering
class QuestionAnsweringTrainerConfig(_TrainerConfigBase):
    pass


@typechecked
class QuestionAnsweringTrainer(_TrainerWithDatasetPreprocessor):
    @override
    def _preprocess_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        inputs, targets = data["inputs"], data["targets"]
        assert len(inputs) == 2 and len(targets) == 1
        assert isinstance(inputs[0], str) and isinstance(inputs[1], str)
        question, context = inputs
        assert isinstance(targets[0], dict)
        assert targets[0].keys() == {"text", "answer_start"}
        text, answer_start = targets[0]["text"], targets[0]["answer_start"]
        assert isinstance(text, list) and isinstance(text[0], str)
        assert isinstance(answer_start, list) and isinstance(answer_start[0], int)
        text, answer_start = text[0], answer_start[0]
        answer_end = answer_start + len(text) - 1  # word offset in sentence

        input_ids: list[int] = [
            self._tokenizer.bos_token_id,
            *self._tokenizer.encode(question, add_special_tokens=False),
            self._tokenizer.sep_token_id,
        ]

        out = self._tokenizer(
            context,
            add_special_tokens=False,
            truncation=False,
        )
        start_position = out.char_to_token(0, answer_start) + len(input_ids)
        end_position = out.char_to_token(0, answer_end) + len(input_ids)
        input_ids += out.input_ids + [self._tokenizer.eos_token_id]
        if len(input_ids) > self._tokenizer.model_max_length:
            return None  # discard long sentences
        return {
            "input_ids": input_ids,
            "start_positions": start_position,
            "end_positions": end_position,
        }

    @override
    def _make_evaluation_metrics(
        self,
    ) -> Callable[[transformers.trainer_utils.EvalPrediction], dict[str, float]] | None:
        def eval_metrics(
            outputs: transformers.trainer_utils.EvalPrediction,
        ) -> dict[str, float]:
            # Int[T, "batch"]
            gt_start, gt_end = outputs.label_ids
            pred_start, pred_end = outputs.predictions
            B = pred_start.shape[0]
            assert len(pred_start.shape) == 1
            assert pred_start.shape == pred_end.shape == gt_start.shape == gt_end.shape
            em = ((pred_start == gt_start) & (pred_end == gt_end)).mean()
            f1 = 0.0
            for ps, pe, gs, ge in zip(pred_start, pred_end, gt_start, gt_end):
                if ps > pe:
                    continue
                pred_span = set(range(ps, pe + 1))
                gt_span = set(range(gs, ge + 1))
                ins = pred_span & gt_span
                if len(ins) == 0:
                    continue
                precision = len(ins) / len(pred_span)
                recall = len(ins) / len(gt_span)
                f1 += 2 * (precision + recall) / (precision + recall)
            return {"exact_match": em, "f1": f1 / B}

        return eval_metrics

    @override
    def _make_preprocess_logits_for_metrics(self):
        def process(logits, labels):
            assert isinstance(logits, tuple) and len(logits) == 2
            assert isinstance(labels, tuple) and len(labels) == 2
            return (logits[0].argmax(-1), logits[1].argmax(-1))

        return process
