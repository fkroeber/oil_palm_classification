import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

# custom loss function
class HingeLoss(torch.nn.Module):
    def __init__(self, squared=False):
        super(HingeLoss, self).__init__()
        self.squared = squared

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        if self.squared:
            hinge_loss = torch.square(hinge_loss)
        return hinge_loss.mean()


# custom rotation transform
class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


# credits: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Initialise early stopping to retrieve best model after patience epochs without increase in validation loss"""

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
