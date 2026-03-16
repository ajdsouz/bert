# Pretraining

- [Pretraining](#pretraining)
  - [Run](#run)
    - [Pretraining](#pretraining-1)
    - [Evaluation](#evaluation)
  - [Add New](#add-new)
    - [Model](#model)
    - [Dataset](#dataset)
    - [Trainer](#trainer)

## Run

### Pretraining

The checkpoints will be saved into `./results/$run_name`

```bash
python $wandb_project/$run_name \
./config/model/google_bert_uncased_L-8_H-512_A-8_mlm.yaml \
./config/training/pretraining_mlm.yaml
```

```bash
python ./cli/train.py $wandb_project/$model_name"_mlm" \
./config/model/$model_name"_mlm.yaml" \
./config/training/pretraining_mlm.yaml
--wandb_group $model_name \
--wandb_tags pretraining
```

### Evaluation

```bash
python ./cli/train.py $wandb_project/"$model_name"_"$task" \
./config/training/$task.yaml \
--pretrained_path ./results/$model_name"_mlm/"$checkpoint \
--wandb_group $model_name \
--wandb_tags $task
```

## Add New

### Model

Check [/src/modeling/interface.py](/src/modeling/interface.py). Each model is definied by:
- base model + base config
- downstream (pretraining) model + config
- downstream (evaluation) model + config

To add a new model, check the example in [/src/modeling/my_bert](src/modeling/my_bert) and [/src/modeling/huggingface](/src/modeling/huggingface):
- Firstly, you need to define a base model:
  - Inherit `_BaseModelConfigBase`, add additional keys required by your base model.
  - Inherit `ModelInterface` and implement the interface of your base model.
- Then, create a downstream model. It's better to implement it as a transition class:
  - Inherit `_ModelForDownstreamConfigBase`. It has a key `base_config` which stores the config of the base model in last step.
  - Inherit `_ModelForDownstreamInterface`. You need to implement two property `_base_model` and `_base_model_hidden_size`.
- After that, define the pretraining model based on the previous:
  - Implement the modules and forward function.
- Finally, after the model got trained, you can use the evaluation heads in [/src/modeling/_head.py](/src/modeling/_head.py) for evaluation.

### Dataset

Check [/src/unified_dataset/interface.py](/src/unified_dataset/interface.py).

Each dataset is definied by a dataset class + config.

HuggingFace dataset is also supported.

### Trainer

Check [/src/trainer/base.py](/src/trainer/base.py). Each trainer is defined by a trainer class + config.

Trainer could be used in both pretraining and downstream evaluation.

Dataset preprocess is also handled by trainer, generally you only need to implement this for a new trainer.
