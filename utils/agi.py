import torch


def transform_224(imgs: torch.Tensor) -> torch.Tensor:
    """
    Upsample MNIST-like images from 28x28 to 224x224 and replicate to 3 channels.

    Notes:
      - This repo's scripts treat image shape as [B, C, W, H].
      - Output is [B, 3, 224, 224].
    """
    assert imgs.ndim == 4, "imgs must be a 4D tensor [B,C,W,H]"
    imgs_224 = torch.nn.functional.interpolate(
        imgs, scale_factor=8, mode="bilinear", align_corners=False
    )
    if imgs_224.shape[1] == 1:
        imgs_224 = torch.cat([imgs_224, imgs_224, imgs_224], dim=1)
    return imgs_224


def ag_distort_28(
    imgs: torch.Tensor,
    threshold: float = 0.0,
    interval: int = 2,
    phase: int = 1,
    direction=(1, 0),
    return_intermediates: bool = False,
):
    """
    (Optional) Low-resolution (28x28) Abutting Grating Illusion generator.
    Kept for compatibility; you said你不需要左侧分支也可以不用这个函数。

    If return_intermediates=True, returns the same keys as ag_distort_224 but at 28x28.
    """
    assert imgs.ndim == 4, "The images must have four dimensions of B,C,W,H."
    B, C, W, H = imgs.shape

    mask_fg = (imgs > threshold).float()
    mask_bg = 1.0 - mask_fg

    device = imgs.device
    w = torch.arange(W, device=device)
    h = torch.arange(H, device=device)
    ww, hh = torch.meshgrid(w, h, indexing="ij")
    idx = (direction[0] * ww + direction[1] * hh) % interval

    gratings_fg_2d = (idx == 0).float()
    gratings_bg_2d = (idx == phase).float()
    gratings_fg = gratings_fg_2d.unsqueeze(0).unsqueeze(0).expand(B, C, W, H).clone()
    gratings_bg = gratings_bg_2d.unsqueeze(0).unsqueeze(0).expand(B, C, W, H).clone()

    masked_gratings_fg = mask_fg * gratings_fg
    masked_gratings_bg = mask_bg * gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg

    if not return_intermediates:
        return ag_image

    return {
        "original_224": imgs,  # 为了和 224 的 key 统一，这里也用同名 key
        "mask_fg": mask_fg,
        "mask_bg": mask_bg,
        "gratings_fg": gratings_fg,
        "gratings_bg": gratings_bg,
        "masked_gratings_fg": masked_gratings_fg,
        "masked_gratings_bg": masked_gratings_bg,
        "ag_image": ag_image,
    }


def ag_distort_224(
    imgs: torch.Tensor,
    threshold: float = 0.0,
    interval: int = 2,
    phase: int = 1,
    direction=(1, 0),
    return_intermediates: bool = False,
):
    """
    High-resolution (224x224) Abutting Grating Illusion generator.

    If return_intermediates=False (default): returns only the final ag_image (same behavior as before).
    If return_intermediates=True: returns a dict of intermediates used in Figure 1(D) right branch:
      - original_224
      - mask_fg / mask_bg
      - gratings_fg / gratings_bg
      - masked_gratings_fg / masked_gratings_bg
      - ag_image
    """
    assert imgs.ndim == 4, "The images must have four dimensions of B,C,W,H."

    # 1) High-res original
    original_224 = transform_224(imgs)
    B, C, W, H = original_224.shape

    # 2) Masks
    mask_fg = (original_224 > threshold).float()
    mask_bg = 1.0 - mask_fg

    # 3) Alternating gratings (vectorized; same result as nested loops)
    device = original_224.device
    w = torch.arange(W, device=device)
    h = torch.arange(H, device=device)
    ww, hh = torch.meshgrid(w, h, indexing="ij")
    idx = (direction[0] * ww + direction[1] * hh) % interval

    gratings_fg_2d = (idx == 0).float()
    gratings_bg_2d = (idx == phase).float()
    gratings_fg = gratings_fg_2d.unsqueeze(0).unsqueeze(0).expand(B, C, W, H).clone()
    gratings_bg = gratings_bg_2d.unsqueeze(0).unsqueeze(0).expand(B, C, W, H).clone()

    # 4) Apply masks and combine
    masked_gratings_fg = mask_fg * gratings_fg
    masked_gratings_bg = mask_bg * gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg

    if not return_intermediates:
        return ag_image

    return {
        "original_224": original_224,
        "mask_fg": mask_fg,
        "mask_bg": mask_bg,
        "gratings_fg": gratings_fg,
        "gratings_bg": gratings_bg,
        "masked_gratings_fg": masked_gratings_fg,
        "masked_gratings_bg": masked_gratings_bg,
        "ag_image": ag_image,
    }
