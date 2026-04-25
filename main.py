import click

from common.config import cfg
from common.logging import setup_logging
from common.types import PipelineMode
from data.gnn_prepare import prepare_gnn_data
from data.prepare import create_dataset
from data.segmentor_crops import prepare_segmentor_crops
from pipelines import get_pipeline
from pipelines.climb.inference import run_climb_inference
from pipelines.hold_classifier.inference import run_full_inference
from pipelines.route.calibrate_color import run_calibration as run_color_calibration
from pipelines.route.inference import run_route_extraction

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


@cli.command("climb-inference")
@click.option("--maskformer-dir", "-m", required=True)
@click.option("--color-weights", "-c", default=None)
@click.option("--type-weights", "-t", default=None)
@click.option("--image-dir", "-d", required=True)
@click.option("--output", "-o", default="results/")
@click.option("--color-model", default="eva02_color")
@click.option("--type-model", default="eva02_type")
@click.option("--use-sam/--no-sam", default=True)
@click.option("--sam-model", default="facebook/sam2.1-hiera-large")
@click.option("--tta/--no-tta", default=True)
@click.option("--score-thr", type=float, default=0.3)
@click.option("--color-temperature", type=float, default=None)
@click.option("--min-area-frac-of-max", type=float, default=0.03)
@click.option("--min-score", type=float, default=0.5)
@click.option("--keep-volumes", is_flag=True, default=False)
@click.option("--max-holds", type=int, default=150)
@click.option("--preview", is_flag=True, default=False)
def climb_inference(maskformer_dir, color_weights, type_weights, image_dir, output, color_model, type_model, use_sam, sam_model, tta, score_thr, color_temperature, min_area_frac_of_max, min_score, keep_volumes, max_holds, preview):
    run_climb_inference(
        maskformer_dir=maskformer_dir,
        image_dir=image_dir,
        output=output,
        color_weights=color_weights,
        color_model_config=color_model,
        type_weights=type_weights,
        type_model_config=type_model,
        use_sam_refine=use_sam,
        sam_model=sam_model,
        use_tta=tta,
        score_thr=score_thr,
        color_temperature=color_temperature,
        min_area_frac_of_max=min_area_frac_of_max,
        min_score=min_score,
        keep_volumes=keep_volumes,
        max_holds=max_holds,
        preview=preview,
    )


@cli.command("extract-route")
@click.option("--predictions", "-p", required=True)
@click.option("--target-color", "-t", required=True)
@click.option("--output", "-o", default="results/")
@click.option("--color-model", default="eva02_color")
@click.option("--core-thr", type=float, default=None)
@click.option("--possible-thr", type=float, default=None)
@click.option("--rejected-strong-other-thr", type=float, default=None)
@click.option("--track-k", type=float, default=None)
@click.option("--weight-color", type=float, default=None)
@click.option("--weight-graph", type=float, default=None)
@click.option("--weight-track", type=float, default=None)
@click.option("--weight-det", type=float, default=None)
@click.option("--graph-radius-factor", type=float, default=None)
@click.option("--propagation-iters", type=int, default=None)
@click.option("--propagation-alpha", type=float, default=None)
@click.option("--propagation-radius-factor", type=float, default=None)
@click.option("--colour-family-voting/--no-colour-family-voting", default=None)
def extract_route_cmd(
    predictions, target_color, output, color_model,
    core_thr, possible_thr, rejected_strong_other_thr, track_k,
    weight_color, weight_graph, weight_track, weight_det,
    graph_radius_factor, propagation_iters, propagation_alpha, propagation_radius_factor,
    colour_family_voting,
):
    run_route_extraction(
        predictions_path=predictions,
        target_color=target_color,
        output=output,
        color_model_config=color_model,
        core_thr=core_thr,
        possible_thr=possible_thr,
        rejected_strong_other_thr=rejected_strong_other_thr,
        track_k=track_k,
        weight_color=weight_color,
        weight_graph=weight_graph,
        weight_track=weight_track,
        weight_det=weight_det,
        graph_radius_factor=graph_radius_factor,
        propagation_iters=propagation_iters,
        propagation_alpha=propagation_alpha,
        propagation_radius_factor=propagation_radius_factor,
        colour_family_voting=colour_family_voting,
    )


@cli.command("calibrate-color")
@click.option("--maskformer-dir", "-m", required=True)
@click.option("--color-weights", "-c", required=True)
@click.option("--crops-root", "-d", required=True)
@click.option("--output", "-o", default="runs/color_calibration.json")
@click.option("--color-model", default="eva02_color")
@click.option("--tta/--no-tta", default=True)
def calibrate_color_cmd(maskformer_dir, color_weights, crops_root, output, color_model, tta):
    from pathlib import Path

    run_color_calibration(
        maskformer_dir=maskformer_dir,
        color_weights=color_weights,
        crops_root=Path(crops_root),
        out_path=Path(output),
        color_model_config=color_model,
        use_tta=tta,
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
