from pathlib import Path
from typing import Any, Callable

import evaluate
import torch
import transformers
from typeguard import typechecked

from src.modeling import ModelInterface, counting_parameters
from src.unified_dataset import UnifiedDatasetInterface
from src.utils import ConfigBase


class _TrainerConfigBase(ConfigBase):
    seed: int
    logging_steps: float | int
    eval_steps: float | int

    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    dataloader_num_workers: int
    bf16: bool

    optim: str
    learning_rate: float
    lr_scheduler_type: str
    warmup_steps: int


# task_name => trainer
_registered_trainer_classes = dict[str, type]()


@typechecked
class TrainerBase:
    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__.startswith("_"):
            return
        assert cls.__name__.endswith("Trainer")
        assert cls.__name__ not in _registered_trainer_classes, "redefined"
        _registered_trainer_classes[cls.__name__] = cls

    def __init__(
        self,
        config: _TrainerConfigBase,
        output_dir: str | Path,
        model: ModelInterface | transformers.PreTrainedModel,
        train_dataset: UnifiedDatasetInterface,
        eval_dataset: UnifiedDatasetInterface,
        tokenizer: transformers.tokenization_utils_tokenizers.TokenizersBackend,
        disable_tqdm: bool,
        dont_save: bool,
    ):
        self.__conf = config
        self._output_dir = Path(output_dir)
        self._tokenizer = tokenizer

        use_cpu = model.device.type == "cpu"
        config_kwargs = {
            k: v
            for k, v in self.conf.to_dict().items()
            if k in _TrainerConfigBase.__annotations__
        }
        assert len(config_kwargs) == 12
        trainer_args = transformers.TrainingArguments(
            output_dir=None if dont_save else self._output_dir,
            data_seed=self.conf.seed,
            use_cpu=use_cpu,
            disable_tqdm=disable_tqdm,
            report_to="wandb",
            dataloader_persistent_workers=self.conf.dataloader_num_workers > 0,
            dataloader_pin_memory=not use_cpu,
            eval_strategy="steps",
            save_strategy="no" if dont_save else "steps",
            save_steps=self.conf.eval_steps,
            save_total_limit=2,  # latest + best
            load_best_model_at_end=not dont_save,
            **config_kwargs,
        )
        self._trainer = transformers.Trainer(
            model=model,
            args=trainer_args,
            train_dataset=self._make_dataset_wrap(train_dataset),
            eval_dataset=self._make_dataset_wrap(eval_dataset),
            data_collator=self._make_data_collator(),
            compute_metrics=self._make_evaluation_metrics(),
            preprocess_logits_for_metrics=self._make_preprocess_logits_for_metrics(),
        )

    @property
    def conf(self):
        return self.__conf

    def _make_dataset_wrap(self, dataset: UnifiedDatasetInterface):
        return dataset

    def _make_data_collator(self) -> Callable[[list[dict[str, Any]]], dict] | None:
        return transformers.DataCollatorWithPadding(self._tokenizer)

    def _make_evaluation_metrics(
        self,
    ) -> Callable[[transformers.trainer_utils.EvalPrediction], dict[str, float]] | None:
        experiment_id = self._output_dir.name
        metric_acc = evaluate.load("accuracy", experiment_id=experiment_id)
        metric_f1 = evaluate.load("f1", experiment_id=experiment_id)

        def eval_metrics(
            outputs: transformers.trainer_utils.EvalPrediction,
        ) -> dict[str, float]:
            labels = outputs.label_ids.reshape(-1)
            preds = outputs.predictions.reshape(-1)
            mask = labels != -100
            labels, preds = labels[mask], preds[mask]
            acc = metric_acc.compute(predictions=preds, references=labels)
            macro_f1 = metric_f1.compute(
                predictions=preds, references=labels, average="macro"
            )
            micro_f1 = metric_f1.compute(
                predictions=preds, references=labels, average="micro"
            )
            return {
                "accuracy": acc["accuracy"],
                "macro_f1": macro_f1["f1"],
                "micro_f1": micro_f1["f1"],
            }

        return eval_metrics

    def _make_preprocess_logits_for_metrics(
        self,
    ) -> Callable[[torch.FloatTensor, torch.LongTensor], torch.LongTensor] | None:
        return lambda logits, labels: logits.argmax(-1)

    def train(self):
        param_size_training, _ = counting_parameters(self._trainer.model)
        print("\n## Model")
        print(self._trainer.model)
        print(f"""
## Overview
Train set size:         {len(self._trainer.train_dataset)}
Validation set size:    {len(self._trainer.eval_dataset)}
Model device:           {self._trainer.model.device}
Model dtype:            {self._trainer.model.dtype}
Number of Parameters:   {param_size_training:,}
""")# fmt:skip
        self._trainer.train()


@typechecked
def load_trainer(
    config: _TrainerConfigBase,
    output_dir: str,
    model: ModelInterface | transformers.PreTrainedModel,
    train_dataset: UnifiedDatasetInterface,
    eval_dataset: UnifiedDatasetInterface,
    tokenizer: transformers.tokenization_utils_tokenizers.TokenizersBackend,
    disable_tqdm: bool,
    dont_save: bool,
) -> TrainerBase:
    assert config.type.endswith("Trainer")
    cls = _registered_trainer_classes[config.type]
    assert issubclass(cls, TrainerBase)
    return cls(
        config=config,
        output_dir=output_dir,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        disable_tqdm=disable_tqdm,
        dont_save=dont_save,
    )
