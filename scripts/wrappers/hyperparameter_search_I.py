import subprocess
import sys
import os
from itertools import product

# script to execute
pyscript = "/home/fkroeber/projects/oil_palm_classification/scripts/nn_train.py"

# define paths
data_path = "data/palm_detection/resampled"
res_path = "results"

# default fixed hyperparams & configs
train_mode = "finetune"
batch_size = "64"
wandb_project = "forests"

# hyperparam configurations for grid search
nets = [
    "densenet",
    "inception",
    "resnet",
    "resnext",
    "wideresnet",
    "senet",
    "max_vit",
    "swin_vit",
]
optimisers = ["SGD", "Adam"]
lrs = ["0.0001", "0.001"]
hyperparams = list(product(nets, optimisers, lrs))

# cross-validation & repeated experiments
seeds = ["40", "41", "42"]
data_dirs = [
    os.path.join(data_path, "split_0"),
    os.path.join(data_path, "split_1"),
    os.path.join(data_path, "split_2"),
]
repetitions = list(product(seeds, data_dirs))

# perform model trainings & evaluations
for i, (net, optimiser, lr) in enumerate(hyperparams):
    for k, (seed, data_dir) in enumerate(repetitions):
        # set save path
        wandb_runname = f"{net}_{optimiser}_{lr}"
        save_path = os.path.join(res_path, "hypersearch_I", f"{wandb_runname}_{k}")
        try:
            # train & evaluate model
            subprocess.check_output(
                [
                    sys.executable,
                    pyscript,
                    "--data_dir",
                    data_dir,
                    "--model_name",
                    net,
                    "--train_mode",
                    train_mode,
                    "--save_path",
                    save_path,
                    "--batch_size",
                    batch_size,
                    "--optimiser",
                    optimiser,
                    "--lr",
                    lr,
                    "--seed",
                    seed,
                    "--wandb_project",
                    wandb_project,
                    "--wandb_runname",
                    wandb_runname,
                ],
                stderr=subprocess.STDOUT,
            )
            # free disk space
            os.remove(os.path.join(save_path, "model_params.pt"))
        except subprocess.CalledProcessError as e:
            print(e.output.decode())
    # info
    print(f"Hyperparameter configuration {i}/{len(hyperparams)} trained & evaluated.")
