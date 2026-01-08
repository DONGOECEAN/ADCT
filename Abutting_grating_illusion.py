# -*- coding: utf-8 -*-
"""
兼容旧结构的 AG 生成（文件名保持 1abutting_grating_illusion.py）
- 在 224×224 上按 hor/ver & 4/8 生成光栅
- 与旧逻辑一致：阈值 0.5 二值化分前景/背景 → 前景铺设phase=0的栅栏，背景铺设phase=interval/2
"""
import torch

def transform_224(imgs: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    if imgs.shape[-1] == 28:
        imgs = F.interpolate(imgs, scale_factor=8, mode='bilinear', align_corners=False)
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1,3,1,1)
    return imgs

def ag_distort_224(imgs: torch.Tensor, threshold: float, interval: int, phase: int, direction=(1,0)):
    if imgs.shape[-1] == 28:
        imgs = transform_224(imgs)
    elif imgs.shape[1] == 1:
        imgs = imgs.repeat(1,3,1,1)
    B,C,H,W = imgs.shape; device = imgs.device
    # 前景/背景二值掩码
    mask_fg = (imgs > threshold).float()
    mask_bg = 1.0 - mask_fg

    # 矢量化生成条纹（yy 对应 H 维，xx 对应 W 维）
    yy = torch.arange(H, device=device).view(H,1).expand(H,W)
    xx = torch.arange(W, device=device).view(1,W).expand(H,W)
    lin = (direction[0]*yy + direction[1]*xx) % interval

    gratings_fg = (lin == 0).float().view(1,1,H,W).expand(B,C,H,W)
    gratings_bg = (lin == phase).float().view(1,1,H,W).expand(B,C,H,W)
    masked_gratings_fg = mask_fg * gratings_fg
    masked_gratings_bg = mask_bg * gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg
    return ag_image