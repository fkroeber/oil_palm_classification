import os
import wandb
from copy import deepcopy

# script to execute
pyscript = "/home/fkroeber/projects/oil_palm_classification/scripts/nn_train_II.py"

# define paths
res_path = "results/hypersearch_II"
data_path = "data/palm_detection/resampled"
data_dirs = [
    os.path.join(data_path, "split_0"),
    os.path.join(data_path, "split_1"),
    os.path.join(data_path, "split_2"),
]

# define general sweep configuration
sweep_config = {
    "program": pyscript,
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "best_acc_val"},
    "run_cap": 50,
    "parameters": {
        "model_name": {"values": ["resnext", "senet", "swin_vit"]},
        "optimiser": {"values": ["SGD", "Adam"]},
        "loss": {"values": ["BCE", "Hinge", "Hinge_squared"]},
        "lr": {"min": 0.00005, "max": 0.001},
        "lr_speed_core": {"min": 0.01, "max": 1.0},
        "train_mode": {"value": "finetune"},
        "batch_size": {"value": 64},
        "max_epochs": {"value": 50},
        "patience": {"value": 10},
        "seed": {"value": 42},
        "save_path": {"value": res_path},
    },
}

# specific configurations for different data-splits
sweep_config_I = deepcopy(sweep_config)
sweep_config_I["name"] = "sweep-split-I"
sweep_config_I["parameters"]["data_dir"] = {"value": data_dirs[0]}

sweep_config_II = deepcopy(sweep_config)
sweep_config_II["name"] = "sweep-split-II"
sweep_config_II["parameters"]["data_dir"] = {"value": data_dirs[1]}

sweep_config_III = deepcopy(sweep_config)
sweep_config_III["name"] = "sweep-split-III"
sweep_config_III["parameters"]["data_dir"] = {"value": data_dirs[2]}

# initialise & run sweeps sequentially
sweep_id = wandb.sweep(sweep_config_I, project="meadows")
wandb.agent(sweep_id)

sweep_id = wandb.sweep(sweep_config_II, project="meadows")
wandb.agent(sweep_id)

sweep_id = wandb.sweep(sweep_config_III, project="meadows")
wandb.agent(sweep_id)
