import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tqdm import tqdm

from common.config import cfg
from common.preprocessing import crop_and_normalize
from common.types import Detection, ImagePredictions, PipelineMode, SegClass
from data.crop_dataset import get_dataset_info
from data.gnn_dataset import build_graph
from models import create_model
from pipelines import register_pipeline
from pipelines.hold_classifier.postprocess import cluster_colors

logger = logging.getLogger(__name__)
SCORE_THRESHOLD = 0.7


def _load_model(model_name: str, weights_path: str, device):
    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _class_aware_nms(boxes, scores, labels, masks, iou_threshold=0.5):
    keep_indices = []
    for cls_id in labels.unique():
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = cls_mask.nonzero(as_tuple=True)[0]
        nms_keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.append(cls_indices[nms_keep])

    if not keep_indices:
        return boxes[:0], scores[:0], labels[:0], masks[:0]

    keep = torch.cat(keep_indices)
    return boxes[keep], scores[keep], labels[keep], masks[keep]


def _classify_crop(classifier, img_tensor, box, crop_size, class_names, device, mask=None):
    crop = crop_and_normalize(img_tensor, box, crop_size, padding=0, mask=mask)
    logits = classifier(crop.unsqueeze(0).to(device))
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    pred_idx = probs.argmax().item()
    prob_dict = {name: round(probs[i].item(), 4) for i, name in enumerate(class_names)}
    return class_names[pred_idx], prob_dict


def run_full_inference(
    segmentor_weights: str,
    image_dir: str,
    output: str,
    color_weights: str = None,
    type_weights: str = None,
    gnn_weights: str = None,
    preview: bool = False,
):
    device = torch.device(cfg.torch.device)
    seg_model_name = "mask_rcnn_hold"
    seg_mcfg = cfg.model_cfg(seg_model_name)

    segmentor = _load_model(seg_model_name, segmentor_weights, device)

    color_classifier = None
    color_names = None
    color_crop_size = 224
    color_padding = 16
    color_use_mask = False
    if color_weights:
        color_model_name = "hold_color_classifier"
        color_mcfg = cfg.model_cfg(color_model_name)
        color_crop_size = color_mcfg["crop_size"]
        color_padding = color_mcfg["crop_padding"]
        color_use_mask = color_mcfg.get("use_mask_channel", False)
        color_classifier = _load_model(color_model_name, color_weights, device)
        color_names = get_dataset_info(color_model_name).class_names

    type_classifier = None
    type_names = None
    type_crop_size = 224
    type_padding = 16
    type_use_mask = False
    if type_weights:
        type_model_name = "hold_type_classifier"
        type_mcfg = cfg.model_cfg(type_model_name)
        type_crop_size = type_mcfg["crop_size"]
        type_padding = type_mcfg["crop_padding"]
        type_use_mask = type_mcfg.get("use_mask_channel", False)
        type_classifier = _load_model(type_model_name, type_weights, device)
        type_names = get_dataset_info(type_model_name).class_names

    gnn_model = None
    gnn_k = 6
    if gnn_weights:
        gnn_mcfg = cfg.model_cfg("color_gnn")
        gnn_k = gnn_mcfg.get("k_neighbors", 6)
        gnn_model = _load_model("color_gnn", gnn_weights, device)

    seg_dataset_root = seg_mcfg["dataset"]
    seg_ann_path = Path(seg_dataset_root) / "train" / "_annotations.coco.json"
    with open(seg_ann_path) as f:
        seg_cats = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    hold_label_ids = {cid for cid, name in seg_cats.items() if name.lower() == SegClass.HOLD}

    test_dir = Path(image_dir)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in extensions)

    to_tensor = T.ToTensor()
    all_predictions = []

    with torch.no_grad():
        for idx, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Inference"):
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            img_tensor = to_tensor(img).to(device)

            seg_pred = segmentor([img_tensor])[0]
            keep = seg_pred["scores"] > SCORE_THRESHOLD
            boxes = seg_pred["boxes"][keep]
            scores_t = seg_pred["scores"][keep]
            masks = seg_pred["masks"][keep] > 0.5
            labels = seg_pred["labels"][keep]

            boxes, scores_t, labels, masks = _class_aware_nms(
                boxes, scores_t, labels, masks, iou_threshold=0.5,
            )

            detections: list[Detection] = []
            for i in range(len(boxes)):
                label_id = labels[i].item()
                is_hold = label_id in hold_label_ids

                det = Detection(
                    box=boxes[i],
                    mask=masks[i].squeeze(0),
                    seg_label=seg_cats.get(label_id, str(label_id)),
                    score=scores_t[i].item(),
                )

                if is_hold:
                    det_mask = masks[i].squeeze(0)

                    if color_classifier is not None:
                        color_mask = det_mask if color_use_mask else None
                        det.color, det.color_probs = _classify_crop(
                            color_classifier, img_tensor, boxes[i],
                            color_crop_size, color_names, device,
                            mask=color_mask,
                        )

                    if type_classifier is not None:
                        type_mask = det_mask if type_use_mask else None
                        det.hold_type, det.type_probs = _classify_crop(
                            type_classifier, img_tensor, boxes[i],
                            type_crop_size, type_names, device,
                            mask=type_mask,
                        )

                detections.append(det)

            if gnn_model is not None and color_classifier is not None:
                hold_dets = [d for d in detections if d.color_probs is not None]
                if len(hold_dets) >= 2:
                    gnn_boxes = torch.stack([d.box for d in hold_dets])
                    gnn_logits = torch.tensor([
                        list(d.color_probs.values()) for d in hold_dets
                    ])
                    gnn_scores = torch.tensor([d.score for d in hold_dets])
                    graph = build_graph(gnn_boxes, gnn_logits, gnn_scores, iw, ih, k=gnn_k)
                    if graph is not None:
                        graph = graph.to(device)
                        refined_logits, route_logits = gnn_model(graph)
                        refined_probs = F.softmax(refined_logits, dim=1).cpu()
                        for j, d in enumerate(hold_dets):
                            pred_idx = refined_probs[j].argmax().item()
                            d.color = color_names[pred_idx]
                            d.color_probs = {
                                name: round(refined_probs[j][ci].item(), 4)
                                for ci, name in enumerate(color_names)
                            }

                        route_pred = (route_logits > 0).cpu()
                        edge_index = graph.edge_index.cpu()
                        for e in range(edge_index.shape[1]):
                            if route_pred[e]:
                                src_det = hold_dets[edge_index[0, e].item()]
                                dst_det = hold_dets[edge_index[1, e].item()]
                                if src_det.color_cluster is None:
                                    src_det.color_cluster = 0
                                dst_det.color_cluster = src_det.color_cluster

            if color_classifier is not None and gnn_model is None:
                detections = cluster_colors(detections)

            img_preds = ImagePredictions(image=img_path.name, detections=detections)
            all_predictions.append(img_preds)

            img_uint8 = (img_tensor * 255).to(torch.uint8).cpu()

            if detections:
                all_masks = torch.stack([d.mask.cpu() for d in detections])
                img_uint8 = draw_segmentation_masks(img_uint8, all_masks, alpha=0.4)

                box_tensor = torch.stack([d.box.cpu() for d in detections])
                label_strs = [d.display_label() for d in detections]
                img_uint8 = draw_bounding_boxes(img_uint8, box_tensor, labels=label_strs, width=2)

            result_img = T.ToPILImage()(img_uint8)
            result_img.save(out_dir / img_path.name)

            if preview and idx == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.imshow(img)
                ax1.set_title("Original")
                ax1.axis("off")
                ax2.imshow(np.array(result_img))
                ax2.set_title(f"Detections ({len(detections)} objects)")
                ax2.axis("off")
                plt.tight_layout()
                preview_path = out_dir / "preview.png"
                plt.savefig(preview_path, dpi=150)
                plt.close()
                logger.info("Preview saved to %s", preview_path)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump([p.to_dict() for p in all_predictions], f, indent=2)

    logger.info("Saved %d results to %s", len(image_paths), out_dir)
    logger.info("Predictions JSON: %s", out_dir / "predictions.json")


@register_pipeline("hold_classifier", PipelineMode.INFERENCE)
def run_inference(model_name: str, weights: str, output: str, image_dir: str, preview: bool = False):
    mcfg = cfg.model_cfg(model_name)
    augment_mode = mcfg.get("augment_mode", "type")

    color_weights = weights if augment_mode == "color" else None
    type_weights = weights if augment_mode == "type" else None

    run_full_inference(
        segmentor_weights=mcfg["segmentor_weights"],
        image_dir=image_dir,
        output=output,
        color_weights=color_weights,
        type_weights=type_weights,
        preview=preview,
    )
