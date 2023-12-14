import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlconfig import register
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import MeanMetric
from tqdm import tqdm
from tqdm import trange

from ..typing import PathLike
from .trainer import Trainer


@register
class MNISTTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        num_classes: int,
    ):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        self.epoch = 1
        self.best_acc = 0

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.validate()
            self.scheduler.step()

            metrics = dict(
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=valid_loss,
                test_acc=valid_acc,
            )
            mlflow.log_metrics(metrics, step=self.epoch)

            format_string = "Epoch: {}/{}, ".format(self.epoch, self.num_epochs)
            format_string += "train loss: {:.4f}, train acc: {:.4f}, ".format(
                train_loss, train_acc
            )
            format_string += "valid loss: {:.4f}, valid acc: {:.4f}, ".format(
                valid_loss, valid_acc
            )
            format_string += "best acc: {:.4f}.".format(self.best_acc)
            tqdm.write(format_string)

    def train(self):
        self.model.train()

        loss_metric = MeanMetric()
        acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes)

        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_metric.update(loss, weight=x.size(0))
            acc_metric.update(output, y)

        return loss_metric.compute().item(), acc_metric.compute().item()

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        loss_metric = MeanMetric()
        acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes)

        for x, y in tqdm(self.valid_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            loss_metric.update(loss, weight=x.size(0))
            acc_metric.update(output, y)

        valid_acc = acc_metric.compute().item()
        if valid_acc > self.best_acc:
            self.best_acc = valid_acc
            self.save_checkpoint("best.pth")

        return loss_metric.compute().item(), valid_acc

    def save_checkpoint(self, f: PathLike):
        self.model.eval()

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_acc": self.best_acc,
        }

        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f: PathLike):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint["best_acc"]
