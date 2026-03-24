from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common.config import cfg


class ImageDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root: str, transform=None):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0


def get_dataloader(root: str, model_name: str, mode: str) -> DataLoader:
    model_cfg = cfg.model_cfg(model_name, mode)
    batch_size = model_cfg.get("batch_size", 32)
    dataset = ImageDataset(root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.torch.num_workers,
    )
