
import os, pickle, glob
from typing import Tuple, List
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

def to_224(img: Tensor, label: int) -> Tuple[Tensor, int]:
    """Bilinear upsample to 224 × 224 with antialias; keeps the label unchanged."""
    img = F.interpolate(
        img.unsqueeze(0),  # (1, C, H, W)
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    return img, label

class IshiharaDataset(Dataset):
    """Memory‑efficient loader for all Ishihara plates.

    Args:
        path (str): root folder, e.g. "./datasets/Ishihara MNIST"
        split (str): "train" or "test"
        plate: "all", single int/str plate id, or list of ids
        transform: callable(img, label) -> Anything
    """

    def __init__(self, path: str, split: str = "train", plate="all", transform=None):
        self.path = path
        self.split = split
        self.transform = transform

        # resolve plate folders
        if plate == "all":
            self.plates = sorted(p for p in os.listdir(path) if p.startswith("X_Plate_"))
        elif isinstance(plate, (str, int)):
            self.plates = [f"X_Plate_{plate}"]
        else:
            self.plates = [f"X_Plate_{p}" if not str(p).startswith("X_Plate_") else str(p) for p in plate]

        self._load_all()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_chw_np(t: Tensor) -> np.ndarray:
        """Convert HWC float32 (0‑1) -> CHW float16 numpy without extra copy."""
        if isinstance(t, torch.Tensor):
            if t.ndim == 3 and t.shape[0] not in {1, 3}:
                t = t.permute(2, 0, 1)       # HWC -> CHW
            return t.to(torch.float16).cpu().numpy()
        raise TypeError("Unsupported tensor type")

    def _load_all(self):
        # gather pickle paths
        pkl_paths: List[str] = []
        for plate_dir in self.plates:
            pkl_paths.extend(
                glob.glob(os.path.join(self.path, plate_dir, "**", f"*{self.split}*.pickle"), recursive=True)
            )

        # pass 1: count samples and cache meta
        meta = []    # [(file, offset, n_imgs)]
        total = 0
        sample_shape = None
        for f in pkl_paths:
            with open(f, "rb") as fh:
                obj = pickle.load(fh)
            tensors = []
            for v in obj["files_dic"].values():
                tensors.extend(v)
            n = len(tensors)
            meta.append((f, total, n))
            total += n
            if sample_shape is None and n:
                sample_shape = self._ensure_chw_np(tensors[0]).shape  # (C,H,W)

        if sample_shape is None:
            raise RuntimeError("No images found")

        c, h, w = sample_shape
        imgs_np = np.empty((total, c, h, w), dtype=np.float16)
        labels_np = np.empty((total,), dtype=np.int64)

        # pass 2: fill arrays in‑place (no big peak)
        for f, offset, _ in meta:
            with open(f, "rb") as fh:
                obj = pickle.load(fh)
            for lbl_str, tlist in obj["files_dic"].items():
                lbl = int(lbl_str)
                for t in tlist:
                    imgs_np[offset] = self._ensure_chw_np(t)
                    labels_np[offset] = lbl
                    offset += 1

        # convert to torch tensors (share memory)
        self.images: Tensor = torch.as_tensor(imgs_np, dtype=torch.float16)
        self.labels: Tensor = torch.as_tensor(labels_np, dtype=torch.long)

        # free numpy refs
        del imgs_np, labels_np

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.images[idx].to(torch.float32)   # keep transforms happy
        label = int(self.labels[idx])
        if self.transform:
            img, label = self.transform(img, label)
        return img, label
