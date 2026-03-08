import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float)
parser.add_argument("--memmap_path", type=str)
parser.add_argument("--activation", type=str)
parser.add_argument("--mlp_type", type=str)
parser.add_argument("--norm_position", type=str)
parser.add_argument("--checkpoint_base_dir", type=str)
parser.add_argument("--log_file_dir", type=str)

args = parser.parse_args()

beta1s = [0.85, 0.9, 0.95]
weight_decays = [0.005, 0.01, 0.015, 0.02]
n_epochs = [1]

for weight_decay in weight_decays:
    for beta1 in beta1s:
        for num_epoch in n_epochs:
            sweep_name = f"lr-{args.lr}-weight_decay-{weight_decay}-beta1-{beta1}-activation-{args.activation}-mlp_type-{args.mlp_type}-norm_position-{args.norm_position}"
            checkpoint_dir = f"{args.checkpoint_base_dir}/{sweep_name}"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print(f"created checkpoint subdirectory at {checkpoint_dir}")

            logfile_path = f"{args.log_file_dir}/{sweep_name}.log"
            if not os.path.exists(logfile_path):
                with open(logfile_path, 'w') as lf:
                    pass
            cmd = [
                "uv", "run", "scripts/train.py",
                "--model", "FacebookAI/roberta-base",
                "--memmap_path", f"{args.memmap_path}",
                "--batch_size", "32",
                "--block_size", "256",
                "--d_model", "512",
                "--d_ffn", "2048",
                "--n_heads", "8",
                "--n_layer", "12",
                "--dropout", "0.1",
                "--vocab_size", "50265",
                "--activation", f"{args.activation}",
                "--mlp_type", f"{args.mlp_type}",
                "--norm_position", f"{args.norm_position}",
                "--lr", f"{args.lr}",
                "--beta1", f"{beta1}",
                "--beta2", "0.98",
                "--checkpoint_dir", f"{checkpoint_dir}",
                "--log_file", f"{logfile_path}",
                "--wandb_entity", "tororo",
                "--wandb_project_name", "Encoder Pretraining Sweeps",
                "--wandb_run_name", f"{sweep_name}",
                "--device", "cuda",
                "--grad_accumulation_steps", "4",
                "--num_epochs", f"{num_epoch}",
                "--save_every", "750",
                "--eval_every", "750",
                "--num_train_tokens", "75000000",
                "--num_val_tokens", "1000000"
            ]

            subprocess.run(cmd, check=True)






