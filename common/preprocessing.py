import torch
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def crop_and_normalize(
    img_tensor: torch.Tensor,
    box: torch.Tensor,
    crop_size: int,
    padding: int,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    _, h, w = img_tensor.shape
    x1, y1, x2, y2 = box.int().tolist()
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = img_tensor[:3, y1:y2, x1:x2]
    crop = T.Resize((crop_size, crop_size), antialias=True)(crop)
    crop = _normalize(crop)

    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2].unsqueeze(0).float()
        mask_crop = T.Resize((crop_size, crop_size), antialias=True)(mask_crop)
        crop = torch.cat([crop, mask_crop], dim=0)

    return crop


def normalize_tensor(
    tensor: torch.Tensor,
    mask_tensor: torch.Tensor = None,
) -> torch.Tensor:
    rgb = tensor[:3]
    rgb = _normalize(rgb)

    if mask_tensor is not None:
        return torch.cat([rgb, mask_tensor], dim=0)

    return rgb
