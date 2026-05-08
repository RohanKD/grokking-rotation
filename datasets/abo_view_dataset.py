import math
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ABOViewPairDataset(Dataset):
    """
    Returns:
      x_source: Tensor [3, H, W], normalized to [-1, 1]
      x_target: Tensor [3, H, W], normalized to [-1, 1]
      delta_angle: scalar degrees
      delta_sin_cos: Tensor [2]
      source_angle: scalar degrees
      target_angle: scalar degrees
      object_id_index: int
      complexity_quartile: int
    """

    def __init__(
        self,
        pairs_csv: str,
        image_size: int = 64,
        normalize: bool = True,
        object_id_to_index: Optional[Dict[str, int]] = None,
    ):
        self.df = pd.read_csv(pairs_csv)
        self.image_size = image_size

        tfms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            tfms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        self.transform = transforms.Compose(tfms)

        object_ids = sorted(self.df["object_id"].astype(str).unique().tolist())
        if object_id_to_index is None:
            self.object_id_to_index = {oid: i for i, oid in enumerate(object_ids)}
        else:
            self.object_id_to_index = object_id_to_index

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            return self.transform(im)

    @staticmethod
    def angle_to_sincos(deg: float):
        rad = math.radians(float(deg))
        return torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        x_source = self._load(r["source_image"])
        x_target = self._load(r["target_image"])

        delta_angle = float(r["delta_angle"])
        source_angle = float(r["source_angle"])
        target_angle = float(r["target_angle"])

        object_id = str(r["object_id"])
        object_index = self.object_id_to_index.get(object_id, -1)

        return {
            "x_source": x_source,
            "x_target": x_target,
            "delta_angle": torch.tensor(delta_angle, dtype=torch.float32),
            "delta_sin_cos": self.angle_to_sincos(delta_angle),
            "source_angle": torch.tensor(source_angle, dtype=torch.float32),
            "target_angle": torch.tensor(target_angle, dtype=torch.float32),
            "object_id": object_id,
            "object_index": torch.tensor(object_index, dtype=torch.long),
            "spin_id": str(r["spin_id"]),
            "complexity_quartile": torch.tensor(int(r["complexity_quartile"]), dtype=torch.long),
            "complexity_score": torch.tensor(float(r["complexity_score"]), dtype=torch.float32),
        }