
"""Memory‑efficient Ishihara plate dataset.

✓ 支持大小写不敏感的 train/test 文件名
✓ 默认常驻 float16，几乎不爆内存
✓ plate="all" | 2 | [2,3,4]
"""
from __future__ import annotations
import os
import glob
import pickle
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from collections.abc import Iterable
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

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

PNGto_224 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.8838, 0.7919, 0.7055] ,
                    [0.1193, 0.1830, 0.2444])
]) #transforms.Normalize([0.5]*3, [0.5]*3)

#inception 的299输入时使用如下：
PNGto_299 = transforms.Compose([
    transforms.Resize((299, 299)),  # 先统一到 299
    transforms.ToTensor(),
    transforms.Normalize([0.8838, 0.7919, 0.7055],
                         [0.1193, 0.1830, 0.2444])
])

def build_plate_dataset(root: str = "./datasets/png224",
                        split: str = "train",
                        plate="all",

                        tfm=PNGto_224):
    """
    plate 取值示例
    ----------------
    "all"                         → 8 个 plate 全部拼接
    "X_Plate_5"                   → 只用 5 号
    ["X_Plate_3", "X_Plate_8"]    → 用 3 号和 8 号
    range(2, 6)                   → 2-5 号（任何可迭代都行）
    """
    root = Path(root)

    # 统一把 plate 变成列表或生成器
    if plate == "all":
        plate_iter = [p.name for p in root.iterdir() if p.name.startswith("X_Plate_")]
    elif isinstance(plate, str):
        plate_iter = [plate]
    elif isinstance(plate, Iterable):
        # 把整数序列等也转成标准文件夹名
        plate_iter = [f"X_Plate_{p}" if not str(p).startswith("X_Plate_") else str(p)
                      for p in plate]
    else:
        raise ValueError("plate 参数应为 'all'、字符串，或可迭代对象。")

    sub_roots = [(root / p / p /split) for p in plate_iter]
    #sub_roots = [root / p / p / f"{split}_images" for p in plate_iter]

    ds_list = [datasets.ImageFolder(str(p), transform=tfm) for p in sub_roots]

    if len(ds_list) == 0:
        raise FileNotFoundError(f"在 {root} 找不到任何符合的 plate 数据。")

    return ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]

class IshiharaDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        plate: Union[str, int, List[Union[str, int]]] = "all",
        transform=None,
        dtype: torch.dtype = torch.float16,
        return_float32: bool = True,
    ):
        """Args
        ----
        path: root folder of dataset, e.g. "./datasets/Ishihara MNIST"
        split: "train" or "test" (case‑insensitive)
        plate: "all", single id (2/"2"), or list like [2,3,4]
        transform: callable(img, label) -> any
        dtype: dtype stored in RAM (torch.float16 recommended)
        return_float32: if True, each __getitem__ returns img.to(torch.float32)
                        set False to keep dtype as stored
        """
        self.path = path
        self.split = split.lower()
        assert self.split in {"train", "test"}, "split must be 'train' or 'test'"

        # resolve plate dirs
        if plate == "all":
            self.plates = sorted(
                p for p in os.listdir(path) if p.startswith("X_Plate_")
            )
        elif isinstance(plate, (str, int)):
            self.plates = [f"X_Plate_{plate}" if not str(plate).startswith("X_Plate_") else str(plate)]
        else:
            self.plates = [
                f"X_Plate_{p}" if not str(p).startswith("X_Plate_") else str(p)
                for p in plate
            ]

        self.transform = transform
        self.dtype = dtype
        self.return_float32 = return_float32

        self._load_all()

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chw_float16_numpy(t: Tensor) -> np.ndarray:
        """Convert HWC float32 0‑1 tensor to CHW float16 numpy array."""
        if t.ndim == 3 and t.shape[0] not in {1, 3}:
            t = t.permute(2, 0, 1)  # HWC -> CHW
        return t.to(torch.float16).cpu().numpy()

    def _find_pickle_files(self) -> List[str]:
        files: List[str] = []
        for plate_dir in self.plates:
            full_dir = os.path.join(self.path, plate_dir)
            # recursively search *.pickle, then filter by split substring (case‑insensitive)
            for f in glob.glob(os.path.join(full_dir, "**", "*.pickle"), recursive=True):
                if self.split in os.path.basename(f).lower():
                    files.append(f)
        return sorted(files)

    def _load_all(self):
        pkl_files = self._find_pickle_files()
        if not pkl_files:
            raise RuntimeError(f"No pickle files found for split='{self.split}' under {self.plates}")

        # pass‑1: count samples & get shape
        total = 0
        sample_shape = None
        meta = []  # (file, offset, n_imgs)
        for f in pkl_files:
            with open(f, "rb") as fh:
                obj = pickle.load(fh)
            n_imgs = 0
            for v in obj["files_dic"].values():
                n_imgs += len(v)
                if sample_shape is None and v:
                    sample_shape = self._chw_float16_numpy(v[0]).shape  # (C,H,W)
            meta.append((f, total, n_imgs))
            total += n_imgs

        if sample_shape is None:
            raise RuntimeError("Failed to determine image shape—dataset seems empty")

        c, h, w = sample_shape
        imgs_np = np.empty((total, c, h, w), dtype=np.float16)
        labels_np = np.empty((total,), dtype=np.int64)

        # pass‑2: fill arrays
        for f, offset, _ in meta:
            with open(f, "rb") as fh:
                obj = pickle.load(fh)
            for lbl_str, tlist in obj["files_dic"].items():
                lbl = int(lbl_str)
                for t in tlist:
                    imgs_np[offset] = self._chw_float16_numpy(t)
                    labels_np[offset] = lbl
                    offset += 1

        # convert to torch tensors (share underlying buffer)
        self.images: Tensor = torch.as_tensor(imgs_np, dtype=self.dtype)
        self.labels: Tensor = torch.as_tensor(labels_np, dtype=torch.long)

        # free numpy references
        del imgs_np, labels_np

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.images[idx]
        if self.return_float32:
            img = img.to(torch.float32)
        label = int(self.labels[idx])
        if self.transform:
            img, label = self.transform(img, label)
        return img, label


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _torch_to_numpy(img: Tensor | np.ndarray) -> np.ndarray:
    """Convert C×H×W tensors/arrays to H×W×C **uint8** ready for imshow."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in {1, 3}:  # C×H×W → H×W×C
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        maxv = float(np.max(img)) if img.size else 0.0
        if maxv <= 1.0:
            img = (img * 255.0).clip(0, 255)
        img = img.astype(np.uint8, copy=False)

    return img


def draw_ishi(image: Tensor | np.ndarray, label: int) -> None:
    img = _torch_to_numpy(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray" if img.ndim == 2 or img.shape[-1] == 1 else None)
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()


def draw_gray_and_color_ishi(
    image1: Tensor | np.ndarray,
    image1_label: int,
    image2: Tensor | np.ndarray,
    image2_label: int,
) -> None:
    img1 = _torch_to_numpy(image1)
    img2 = _torch_to_numpy(image2)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray" if img1.ndim == 2 or img1.shape[-1] == 1 else None)
    plt.title(f"Colour – Label {image1_label}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray" if img2.ndim == 2 or img2.shape[-1] == 1 else None)
    plt.title(f"Grayscale – Label {image2_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
