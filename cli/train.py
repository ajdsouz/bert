import argparse
from pathlib import Path

import torch
import transformers
import datasets
import wandb
import yaml

from src.utils import load_configs_from_yaml_files, make_config_from_dict
from src.modeling import load_model
from src.unified_dataset import load_unified_dataset
from src.tokenizer import load_tokenizer
from src.trainer import load_trainer


def load_configs(config_paths: list[str], pretrained_path: str | Path | None):
    dict_ = dict[str, dict]()
    for p in config_paths:
        with open(p, encoding="utf-8") as file:
            data = yaml.safe_load(file)
        assert len(dict_.keys() & data.keys()) == 0
        dict_.update(data)

    if "model" in dict_ or pretrained_path is None:
        assert dict_.keys() == {"model", "dataset", "trainer"}
        return {k: make_config_from_dict(v) for k, v in dict_.items()}

    assert dict_.keys() == {"_model_patch", "dataset", "trainer"}
    pretrained_config = load_configs_from_yaml_files(
        [Path(pretrained_path).parent / "experiment_setting.yaml"]
    )
    assert pretrained_config.keys() == {"model", "dataset", "trainer"}
    pretrained_config: dict = pretrained_config["model"].base_config.to_dict()

    assert dict_["trainer"]["type"].endswith("Trainer")
    task: str = dict_["trainer"]["type"][:-7]
    assert pretrained_config["type"].endswith("Model")

    downstream_model_config = {
        "type": f"{pretrained_config['type']}For{task}",
        "base_config": pretrained_config,
    }
    model_patch = dict_.pop("_model_patch")
    assert len(model_patch.keys() & downstream_model_config.keys()) == 0
    downstream_model_config.update(model_patch)
    dict_["model"] = downstream_model_config
    return {k: make_config_from_dict(v) for k, v in dict_.items()}


def main(args: argparse.Namespace):
    project_name, experiment_name = str(args.project_experiment_name).split("/")
    output_dir = Path("./results") / experiment_name

    if args.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()
        datasets.utils.logging.disable_progress_bar()

    # load experiment settings and setup random seed
    configs = load_configs(args.config_paths, args.pretrained_path)
    torch.manual_seed(configs["trainer"].seed)
    torch.cuda.manual_seed_all(configs["trainer"].seed)
    transformers.set_seed(configs["trainer"].seed)

    # load models
    model = load_model(configs["model"], pretrained_path=args.pretrained_path)
    dataset = load_unified_dataset(
        configs["dataset"], "./data", {"train", "validation"}
    )
    if args.pretrained_path:
        tokenizer = load_tokenizer(args.pretrained_path)
    else:
        tokenizer = load_tokenizer(dataset["train"].conf.tokenizer)
    assert getattr(model.config, "base_config", model.config).vocab_size == len(tokenizer) # fmt:skip

    # print and save experiment settings
    print("# Masked Language Model Pretraining:", experiment_name)
    configs_text, fingerprints = list[str](), dict[str, str]()
    for k, v in configs.items():
        fp = f"{v.fingerprint():064x}"
        text = v.to_yaml(with_root_key=k).replace(f"{k}:\n", f"{k}: # {fp}\n")
        configs_text.append(text)
        fingerprints[k] = fp
    configs_text = "".join(configs_text)
    print(f"## Configs\n{configs_text}")
    if not args.dont_save:
        output_dir.mkdir(parents=True, exist_ok=False)
        with open(
            output_dir / "experiment_setting.yaml", "w", encoding="utf-8"
        ) as file:
            file.write(configs_text)

    # train
    wandb_config = {"experiment_setting": configs, "_fingerprints": fingerprints}
    print(f"## Wandb")
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=wandb_config,
        group=args.wandb_group,
        tags=args.wandb_tags,
    )
    trainer = load_trainer(
        config=configs["trainer"],
        output_dir=str(output_dir),
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        disable_tqdm=args.disable_tqdm,
        dont_save=args.dont_save,
    )
    trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_experiment_name", type=str)
    parser.add_argument("config_paths", type=str, nargs="+")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--dont_save", action="store_true", default=False)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
