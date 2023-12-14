from pathlib import Path

import click
import mlconfig
import mlflow
import torch
from mlconfig import instantiate
from omegaconf import OmegaConf

from ..cli import cli
from ..utils import manual_seed


@cli.command()
@click.option(
    "-c", "--config-file", type=click.Path(path_type=Path), default="configs/mnist.yaml"
)
@click.option("-r", "--resume", type=click.Path(path_type=Path), default=None)
def train(config_file: Path, resume: Path):
    config = mlconfig.load(config_file)

    mlflow.log_text(OmegaConf.to_yaml(config), artifact_file="config.yaml")

    mlflow.log_params(config.log_params)

    manual_seed()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = instantiate(config.model).to(device)
    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.scheduler, optimizer)
    train_loader = instantiate(config.train_loader)
    valid_loader = instantiate(config.valid_loader)
    # test_loader = instantiate(config.test_loader)

    trainer = instantiate(
        config.trainer,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    if resume is not None:
        trainer.resume(resume)

    trainer.fit()
