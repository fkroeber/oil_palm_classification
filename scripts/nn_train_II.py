import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn
import wandb

from nn_models import NN
from nn_utils import EarlyStopping, HingeLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Trainer:
    """NN training & evaluation on a data set of choice"""

    def __init__(self, **kwargs):
        """
        Define model training & logging parameters
        data_dir: data path with subfolders "train", "val" & "test" with structure of each corresponding to torchvision.datasets.ImageFolder
        model_name: model to be trained - see "nn_models" for list of available args (except Inception)
        train_mode: degree of use of pretrained models - see "nn_models" for list of available args
        batch_size: batch size used for training - choose a multiple of 2
        optimiser: optimiser used for training - available optimisers are "SGD", "Adam"
        loss: loss function used for training - options are "BCE", "Hinge" and "Hinge_squared"
        lr: learning rate used for training
        lr_speed_core: learning rate relative to lr used for training of the pre-trained core part of the net
        max_epochs: maximum number of epochs used for training
        patience: patience for early stopping on validation data
        seed: optional seed to reproduce results
        save_path: path to save results for model run
        wandb_project: project name for logging results with wandb
        wandb_runname: name current run logged with wandb
        """
        self.data_dir = kwargs.get("data_dir")
        self.model_name = kwargs.get("model_name")
        self.train_mode = kwargs.get("train_mode")
        self.batch_size = kwargs.get("batch_size")
        self.optimiser = kwargs.get("optimiser")
        self.loss = kwargs.get("loss")
        self.lr = kwargs.get("lr")
        self.lr_speed_core = kwargs.get("lr_speed_core")
        self.max_epochs = kwargs.get("max_epochs")
        self.patience = kwargs.get("patience")
        self.seed = kwargs.get("seed")
        self.save_path = os.path.join(
            kwargs.get("save_path"),
            f"{self.model_name}_{self.loss}_{self.optimiser}_{round(self.lr, 5)}",
        )
        self.wandb_project = kwargs.get("wandb_project")
        self.wandb_runname = f"{self.model_name}_{self.loss}_{self.optimiser}"
        # create save folder if necessary
        os.makedirs(self.save_path, exist_ok=True)
        # set seed if specified
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        # define wandb session & log model configuration
        wandb.init(
            project=self.wandb_project,
            dir=self.save_path,
        )
        wandb.run.name = self.wandb_runname
        self._update_logs()

    def _update_logs(self):
        """
        Write config logs to disk & update WandB cloud logs
        """
        configs = {
            k: v for k, v in self.__dict__.items() if type(v) in [str, int, float, bool]
        }
        pd_configs = pd.DataFrame.from_dict(data=configs, orient="index")
        pd_configs.to_csv(
            os.path.join(self.save_path, "model_config.csv"), header=False
        )
        wandb.config.update(configs)

    def prep_model(self):
        """
        Intitialise model and dataloaders
        """
        # get number of classes for final fc layer
        num_classes = len(
            datasets.ImageFolder(os.path.join(self.data_dir, "train")).classes
        )
        # init model & transforms
        model_kwargs = {
            "model_name": self.model_name,
            "mode": self.train_mode,
            "num_classes": 1 if num_classes == 2 else num_classes,
            "seed": self.seed,
        }
        model_ft, model_transform = NN(**model_kwargs).init_model()
        data_transforms = {
            "train": transforms.Compose(
                [
                    model_transform,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            ),
            "val": model_transform,
            "test": model_transform,
        }
        # create training and validation dataloaders
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x])
            for x in ["train", "val", "test"]
        }
        loader_args = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,  # speed-up host-to-device transfer
        }
        self.dataloaders = {
            x: DataLoader(image_datasets[x], **loader_args)
            for x in ["train", "val", "test"]
        }
        self.size_train_set = len(self.dataloaders["train"].dataset)
        self.size_val_set = len(self.dataloaders["val"].dataset)
        self.size_test_set = len(self.dataloaders["test"].dataset)
        # set model & calculate number of parameters
        self.model_ft = model_ft
        self.total_params = sum(p.numel() for p in model_ft.parameters())
        self.trainable_params = sum(
            p.numel() for p in model_ft.parameters() if p.requires_grad
        )
        self._update_logs()

    # credits: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    def train_model(self):
        """Train model"""
        # check GPU availability & send the model to GPU
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        model_ft = self.model_ft.to(device)
        # gather params to be updated
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        # set up optimiser
        if self.optimiser == "SGD":
            optimizer_ft = torch.optim.SGD(
                [
                    {
                        "params": list(model_ft.children())[:-1][0].parameters(),
                        "lr": self.lr * self.lr_speed_core,
                    },
                    {
                        "params": list(model_ft.children())[-1].parameters(),
                        "lr": self.lr,
                    },
                ],
                momentum=0.9,
            )
        elif self.optimiser == "Adam":
            optimizer_ft = torch.optim.Adam(
                [
                    {
                        "params": list(model_ft.children())[:-1][0].parameters(),
                        "lr": self.lr * self.lr_speed_core,
                    },
                    {
                        "params": list(model_ft.children())[-1].parameters(),
                        "lr": self.lr,
                    },
                ]
            )
        # set up loss
        if self.loss == "BCE":
            criterion = nn.BCEWithLogitsLoss()
        elif self.loss == "Hinge":
            criterion = HingeLoss()
        elif self.loss == "Hinge_squared":
            criterion = HingeLoss(squared=True)
        # initialise early stopping
        early_stopping = EarlyStopping(
            patience=self.patience, path=os.path.join(self.save_path, "model_params.pt")
        )
        # create logging objects
        self.loss_history = {"train": [], "val": [], "test": []}
        self.acc_history = {"train": [], "val": [], "test": []}

        # start training
        since = time.time()
        for epoch in range(self.max_epochs):
            if early_stopping.early_stop:
                break
            print("Epoch {}/{}".format(epoch, self.max_epochs - 1))
            print("-" * 10)
            for phase in ["train", "val", "test"]:
                if phase == "train":
                    model_ft.train()
                else:
                    model_ft.eval()
                running_loss = 0.0
                running_corrects = 0
                # iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.float)
                    # zero the parameter gradients
                    optimizer_ft.zero_grad()
                    # forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model_ft(inputs).view(-1)
                        if self.loss in ["Hinge", "Hinge_squared"]:
                            # convert outputs & targets to range [-1, 1]
                            outputs_ = nn.Tanh()(outputs)
                            labels_ = torch.clone(labels)
                            labels_[labels_ == 0] = -1
                            # calculate loss & preds
                            loss = criterion(outputs_, labels_)
                        elif self.loss == "BCE":
                            # calculate loss & preds
                            loss = criterion(outputs, labels)
                        preds = torch.round(torch.sigmoid(outputs))
                    # backward pass only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer_ft.step()
                    # add iteration statistics to epoch summary
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
                # calculate & log epoch loss & accs
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects / len(self.dataloaders[phase].dataset)
                self.acc_history[phase].append(epoch_acc)
                self.loss_history[phase].append(epoch_loss)
                wandb.log(
                    {
                        f"loss_{phase}": epoch_loss,
                        f"acc_{phase}": epoch_acc,
                    },
                    step=epoch,
                )
                print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")
                if phase == "test":
                    print()
                # evaluate early stopping
                if phase == "val":
                    early_stopping(epoch_loss, model_ft)

        # load best model
        model_ft.load_state_dict(
            torch.load(os.path.join(self.save_path, "model_params.pt"))
        )
        self.model_ft = model_ft
        # update logs
        self.training_epochs = epoch
        self.training_time_total = round(time.time() - since, 2)
        self.training_time_epoch = round(
            self.training_time_total / self.training_epochs, 2
        )
        pd.DataFrame(self.loss_history).to_csv(
            os.path.join(self.save_path, "model_loss.csv"), index=False
        )
        pd.DataFrame(self.acc_history).to_csv(
            os.path.join(self.save_path, "model_acc.csv"), index=False
        )
        self._update_logs()
        # set summary vals based on early stopping point
        if self.patience < self.max_epochs:
            val_idx_min = np.argmin(self.loss_history["val"])
            for metric in ["acc", "loss"]:
                for phase in ["train", "val", "test"]:
                    final_val = eval(f"self.{metric}_history['{phase}'][{val_idx_min}]")
                    wandb.run.summary[f"best_{metric}_{phase}"] = final_val
        wandb.finish()


if __name__ == "__main__":

    # parse arguments
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Train & evaluate neural network for classification",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data path with structure corresponding to torchvision.datasets.ImageFolder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model architecture to be trained, see nn_models.py for avaliable models",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        help="degree of pretraining - see nn_models.py for available args",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save results for model run (model & logs)",
    )
    parser.add_argument("--batch_size", type=int, help="size of the batch")
    parser.add_argument(
        "--optimiser",
        type=str,
        choices=["SGD", "Adam"],
        help="steepest gradient algorithm",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["BCE", "Hinge", "Hinge_squared"],
        help="loss function for classification",
    )
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument(
        "--lr_speed_core",
        type=float,
        help="relative lr for adjusting weights of core part of the pre-trained net",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="maximal number of epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="patience for early stopping on validation data",
    )
    parser.add_argument(
        "--seed", nargs="?", type=int, default=None, help="seed to reproduce results"
    )
    parser.add_argument(
        "--wandb_project", type=str, help="project name for logging results with WandB"
    )
    parser.add_argument(
        "--wandb_runname", type=str, help="name current run logged with WandB"
    )

    # re-parse booleans correctly
    config = vars(parser.parse_args())

    # train & evaluate model
    trainer = Trainer(**config)
    trainer.prep_model()
    trainer.train_model()
