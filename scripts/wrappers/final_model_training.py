import subprocess
import sys
import os
from itertools import product

# script to execute
pyscript = "/home/fkroeber/projects/oil_palm_classification/scripts/nn_train_II.py"

# define paths
wandb_project = "fields"
data_path = "data/palm_detection/resampled"
res_path = "results"

# default fixed hyperparams & configs
net = "senet"
train_mode = "finetune"
batch_size = "64"
optimiser = "Adam"
loss = "Hinge_squared"
lr = "0.0005"
lr_speed_core = "0.5"

# cross-validation & repeated experiments
seeds = ["42"]
data_dirs = [
    os.path.join(data_path, "split_0"),
    os.path.join(data_path, "split_1"),
    os.path.join(data_path, "split_2"),
]
repetitions = list(product(seeds, data_dirs))

# perform model trainings & evaluations
for i, net in enumerate([net]):
    for k, (seed, data_dir) in enumerate(repetitions):
        # set save path
        wandb_runname = f"{net}_{os.path.split(data_dir)[-1]}"
        save_path = os.path.join(res_path, "final_model", f"{wandb_runname}_{k}")
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
                    "--loss",
                    loss,
                    "--lr",
                    lr,
                    "--lr_speed_core",
                    lr_speed_core,
                    "--seed",
                    seed,
                    "--wandb_project",
                    wandb_project,
                    "--wandb_runname",
                    wandb_runname,
                ],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(e.output.decode())
