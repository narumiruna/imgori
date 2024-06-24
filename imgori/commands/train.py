from pathlib import Path

import click
import mlconfig
import torch
import wandb
from mlconfig import flatten
from mlconfig import instantiate
from omegaconf import OmegaConf

from ..cli import cli
from ..utils import manual_seed


@cli.command()
@click.option("-c", "--config-file", type=click.Path(path_type=Path), default="configs/mnist.yaml")
@click.option("-r", "--resume", type=click.Path(path_type=Path), default=None)
def train(config_file: Path, resume: Path):
    wandb.login()

    with wandb.init():
        wandb.save(config_file)
        config = mlconfig.load(config_file)

        wandb.config.update(flatten(OmegaConf.to_object(config)))

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
