import click

from common.config import cfg
from pipelines import get_pipeline


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_name")
@click.option("--output", "-o", required=True)
def train(model_name, output):
    pipeline_name = cfg.models[model_name]["pipeline"]
    run = get_pipeline(pipeline_name, "train")
    run(model_name, output)


@cli.command()
@click.argument("model_name")
@click.option("--output", "-o", required=True)
def validate(model_name, output):
    pipeline_name = cfg.models[model_name]["pipeline"]
    run = get_pipeline(pipeline_name, "validate")
    run(model_name, output)


@cli.command()
@click.argument("model_name")
@click.option("--weights", "-w", required=True)
@click.option("--output", "-o", default="results/")
def inference(model_name, weights, output):
    pipeline_name = cfg.models[model_name]["pipeline"]
    run = get_pipeline(pipeline_name, "inference")
    run(model_name, weights, output)


if __name__ == "__main__":
    cli()
