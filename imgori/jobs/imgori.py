import mlflow
import torch
from mlconfig import instantiate
from mlconfig import register
from omegaconf import OmegaConf

from ..typing import PathLike
from ..utils import manual_seed
from .job import Job


@register
class ImgoriTrainJob(Job):
    def run(self, config: OmegaConf, resume: PathLike = None) -> None:
        mlflow.log_text(OmegaConf.to_yaml(config), artifact_file="config.yaml")

        mlflow.log_params(config.log_params)

        manual_seed()

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = instantiate(config.model).to(device)
        optimizer = instantiate(config.optimizer, model.parameters())
        scheduler = instantiate(config.scheduler, optimizer)
        train_loader = instantiate(config.dataset, phase="train")
        valid_loader = instantiate(config.dataset, phase="valid")

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
