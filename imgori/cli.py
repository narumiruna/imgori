from pathlib import Path

import click
from mlconfig import instantiate
from mlconfig import load


@click.command()
@click.option(
    "-c", "--config-file", type=click.Path(path_type=Path), default="configs/mnist.yaml"
)
@click.option("-r", "--resume", type=click.Path(path_type=Path), default=None)
def main(config_file: Path, resume: Path):
    config = load(config_file)

    job = instantiate(config.job)
    job.run(config, resume)
