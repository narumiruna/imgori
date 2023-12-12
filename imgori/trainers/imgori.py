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
class ImgoriTrainer(Trainer):
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
    ) -> None:
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        self.epoch = 1
        self.metrics = dict(best_acc=0)

    def fit(self) -> None:
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            self.train()
            self.validate()
            self.scheduler.step()

            mlflow.log_metrics(self.metrics, step=self.epoch)

            format_string = "Epoch: {}/{}".format(self.epoch, self.num_epochs)
            for k, v in self.metrics.items():
                format_string += ", {}: {:.4f}".format(k, v)
            tqdm.write(format_string)

    def train(self) -> None:
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

        self.metrics.update(
            train_loss=loss_metric.compute().item(),
            train_acc=acc_metric.compute().item(),
        )

    @torch.no_grad()
    def validate(self) -> None:
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
        if valid_acc > self.metrics["best_acc"]:
            self.metrics["best_acc"] = valid_acc
            self.save_checkpoint("best.pth")

        self.metrics.update(
            valid_loss=loss_metric.compute().item(),
            valid_acc=valid_acc,
        )

    def save_checkpoint(self, f: PathLike) -> None:
        self.model.eval()

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "metrics": self.metrics,
        }

        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f: PathLike) -> None:
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epoch = checkpoint["epoch"] + 1
        self.metrics = checkpoint["metrics"]
