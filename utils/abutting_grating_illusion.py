import torch
from torchvision import datasets, transforms
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import torch.nn.functional as F
def ag_distort_28(imgs, threshold=0, interval=2, phase=1, direction=(1,0)):
    #return imgs
    assert len(imgs.shape) == 4, "The images must have four dimensions of B,C,W,H."
    B,C,W,H = imgs.shape
    mask_fg = (imgs>threshold).float()  
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
  
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==phase:
                gratings_bg[:,:,w,h] = 1
    masked_gratings_fg = mask_fg*gratings_fg
    masked_gratings_bg = mask_bg*gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg
    return ag_image

def transform_224(imgs):
    imgs = torch.nn.functional.interpolate(imgs, scale_factor = 8, mode = 'bilinear', align_corners = False)
    imgs = torch.cat([imgs, imgs, imgs], dim=1)
    return imgs   

def ag_distort_224(imgs, threshold=0, interval=2, phase=1, direction=(1,0)):
    assert len(imgs.shape) == 4, "The images must have four dimensions of C,W,H."   
    imgs = torch.nn.functional.interpolate(imgs, scale_factor = 8, mode = 'bilinear', align_corners = False)
    imgs = torch.cat([imgs, imgs, imgs], dim=1)
    #return imgs
    B,C,W,H = imgs.shape
    mask_fg = (imgs>threshold).float()  
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
  
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==phase:
                gratings_bg[:,:,w,h] = 1
    masked_gratings_fg = mask_fg*gratings_fg
    masked_gratings_bg = mask_bg*gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg
    return ag_image



def transform_299(imgs, force_cpu=False):
    orig_dev = imgs.device
    dev = torch.device('cpu') if force_cpu else orig_dev
    if imgs.device != dev:
        imgs = imgs.to(dev)

    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)  # 或 imgs = imgs.expand(-1,3,-1,-1).contiguous()
    elif imgs.shape[1] != 3:
        raise ValueError("Expected 1 or 3 channels for MNIST->RGB.")

    if imgs.device != orig_dev and not force_cpu:
        imgs = imgs.to(orig_dev)
    return imgs

def ag_distort_299(imgs, threshold=0.0, interval=2, phase=1, direction=(1, 0), force_cpu=False):
    assert imgs.ndim == 4, "Expect [B, C, H, W]"
    if interval <= 0:
        raise ValueError("interval must be positive")

    orig_dev = imgs.device
    dev = torch.device('cpu') if force_cpu else orig_dev
    if imgs.device != dev:
        imgs = imgs.to(dev)

    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)

    B, C, H, W = imgs.shape
    ys = torch.arange(H, device=dev).view(H, 1)
    xs = torch.arange(W, device=dev).view(1, W)
    grid = (direction[1] * xs + direction[0] * ys) % interval
    phase_mod = phase % interval

    grat_fg = (grid == 0).float().unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    grat_bg = (grid == phase_mod).float().unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    mask_fg = (imgs > threshold).float()
    mask_bg = 1.0 - mask_fg
    ag_image = mask_fg * grat_fg + mask_bg * grat_bg

    if imgs.device != orig_dev and not force_cpu:
        ag_image = ag_image.to(orig_dev)
    return ag_image

#inceptionV3网络用：
'''def transform_299(imgs):
    """
    imgs: [B, C, H, W]，通常 MNIST 为 C=1
    输出: [B, 3, 299, 299]
    """
    # 直接插值到 299×299
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    # 灰度变三通道
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)   # 或者 imgs = imgs.expand(-1, 3, -1, -1)
    elif imgs.shape[1] != 3:
        raise ValueError("Expected 1 or 3 channels for MNIST->RGB.")
    return imgs


def ag_distort_299(imgs, threshold=0.0, interval=2, phase=1, direction=(1, 0)):
    """
    在 299×299 分辨率上施加 AG 光栅干扰。
    imgs: [B, C, H, W]（MNIST 通常 C=1）
    返回: [B, 3, 299, 299] 的干扰图
    """
    assert imgs.ndim == 4, "The images must have shape [B, C, H, W]."
    imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    imgs = torch.cat([imgs, imgs, imgs], dim=1)
    # return imgs
    B, C, W, H = imgs.shape
    mask_fg = (imgs > threshold).float()
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)

    for w in range(W):
        for h in range(H):
            if (direction[0] * w + direction[1] * h) % interval == 0:
                gratings_fg[:, :, w, h] = 1
            if (direction[0] * w + direction[1] * h) % interval == phase:
                gratings_bg[:, :, w, h] = 1
    masked_gratings_fg = mask_fg * gratings_fg
    masked_gratings_bg = mask_bg * gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg
    return ag_image

    if not isinstance(interval, int) or interval <= 0:
        raise ValueError("`interval` must be a positive integer.")

    # 先插值到 299×299，再转三通道
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    elif imgs.shape[1] != 3:
        raise ValueError("Expected 1 or 3 channels for MNIST->RGB.")

    B, C, H, W = imgs.shape
    device = imgs.device

    # 前景/背景掩码（按阈值分离）
    mask_fg = (imgs > threshold).float()
    mask_bg = 1.0 - mask_fg

    # 向量化生成光栅（避免双重 for 循环）
    ys = torch.arange(H, device=device).view(H, 1)   # [H,1]
    xs = torch.arange(W, device=device).view(1, W)   # [1,W]
    grid = (direction[1] * xs + direction[0] * ys) % interval  # [H,W]
    phase_mod = phase % interval

    grat_fg = (grid == 0).float().unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    grat_bg = (grid == phase_mod).float().unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    ag_image = mask_fg * grat_fg + mask_bg * grat_bg
    return ag_image'''


def ag_distort_silhouette(imgs, threshold=0.5, interval=2, phase=1, direction=(1,0)):
    assert len(imgs.shape) == 4, "The image must have only three dimensions of C,W,H."
    #imgs = torch.nn.functional.interpolate(imgs, scale_factor = 2, mode = 'bilinear', align_corners = False)
    B,C,W,H = imgs.shape
    mask_fg = (imgs<threshold).float()
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==phase:
                gratings_bg[:,:,w,h] = 1
    ag_images = mask_fg*gratings_fg + mask_bg*gratings_bg
    #transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #transform = transforms.Compose([]) 
    #ag_images[0] = transform(ag_images[0])
    return ag_images

