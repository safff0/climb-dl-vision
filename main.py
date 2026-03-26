import click

from common.config import cfg
from common.logging import setup_logging
from common.types import PipelineMode
from data.prepare import create_dataset
from pipelines import get_pipeline

setup_logging()


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_name")
@click.option("--output", "-o", required=True)
def train(model_name, output):
    pipeline_name = cfg.model_cfg(model_name)["pipeline"]
    run = get_pipeline(pipeline_name, PipelineMode.TRAIN)
    run(model_name, output)


@cli.command()
@click.argument("model_name")
@click.option("--weights", "-w", required=True)
def validate(model_name, weights):
    pipeline_name = cfg.model_cfg(model_name)["pipeline"]
    run = get_pipeline(pipeline_name, PipelineMode.VALIDATE)
    run(model_name, weights)


@cli.command()
@click.argument("model_name")
@click.option("--weights", "-w", required=True)
@click.option("--image-dir", "-d", required=True)
@click.option("--output", "-o", default="results/")
@click.option("--preview", is_flag=True, default=False)
def inference(model_name, weights, image_dir, output, preview):
    pipeline_name = cfg.model_cfg(model_name)["pipeline"]
    run = get_pipeline(pipeline_name, PipelineMode.INFERENCE)
    run(model_name, weights, output, image_dir=image_dir, preview=preview)


@cli.command("create-dataset")
@click.argument("dataset_name")
@click.option("--url", "-u", required=True)
def create_dataset_cmd(dataset_name, url):
    create_dataset(dataset_name, url)


if __name__ == "__main__":
    cli()
