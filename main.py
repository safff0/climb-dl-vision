import click

from common.config import cfg
from common.logging import setup_logging
from common.types import PipelineMode
from data.gnn_prepare import prepare_gnn_data
from data.prepare import create_dataset
from data.segmentor_crops import prepare_segmentor_crops
from pipelines import get_pipeline
from pipelines.hold_classifier.inference import run_full_inference

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


@cli.command("full-inference")
@click.option("--segmentor-weights", "-s", required=True)
@click.option("--color-weights", "-c", default=None)
@click.option("--type-weights", "-t", default=None)
@click.option("--gnn-weights", "-g", default=None)
@click.option("--handcrafted-color-weights", "-hc", default=None)
@click.option("--image-dir", "-d", required=True)
@click.option("--output", "-o", default="results/")
@click.option("--preview", is_flag=True, default=False)
def full_inference(segmentor_weights, color_weights, type_weights, gnn_weights, handcrafted_color_weights, image_dir, output, preview):
    run_full_inference(
        segmentor_weights=segmentor_weights,
        image_dir=image_dir,
        output=output,
        color_weights=color_weights,
        type_weights=type_weights,
        gnn_weights=gnn_weights,
        handcrafted_color_weights=handcrafted_color_weights,
        preview=preview,
    )


@cli.command("prepare-gnn-data")
@click.argument("model_name")
def prepare_gnn_data_cmd(model_name):
    prepare_gnn_data(model_name)


@cli.command("prepare-crops")
@click.argument("model_name")
def prepare_crops(model_name):
    prepare_segmentor_crops(model_name)


@cli.command("create-dataset")
@click.argument("dataset_name")
@click.option("--url", "-u", required=True)
def create_dataset_cmd(dataset_name, url):
    create_dataset(dataset_name, url)


if __name__ == "__main__":
    cli()
