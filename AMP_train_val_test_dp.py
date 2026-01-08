"""
- 在线 28→224 插值、AG 生成
- Ishihara PNG 目录读取 + 每类限量采样（对齐论文：train 1000 / test 200）
- 三种策略：S0（仅Ish），S1（Ish+AG），S2（S1 + DeepAugment + AugMix）
- 验证集：Ish_val（每类100） + AG6(hor/ver) 作为代理 OOD；用其 mean_acc 选择 best.ckpt
- 训练中每轮仍然测 6 个测试集（仅记录曲线，不用于选择）
- 训练结束后加载 best.ckpt 在 6 个测试集做一次“最终评测”，输出 acc/NLL/ECE/PRF1/CM（可选AUROC）
"""
import os, time, argparse, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
from Ishihara_data import (
    build_ishihara_png, build_ishihara_train_val, collate_and_to224,
    mnist_train_dataset, mnist_test_dataset, normalize_imagenet)
from Abutting_grating_illusion import ag_distort_224, transform_224
from utils_legacy import (augmix_batch, jsd_loss, Noise2NetDA, save_image,
    reliability_points, negative_log_likelihood, expected_calibration_error,
    confusion_matrix_from_logits, prf1_from_confusion_matrix, auroc_one_vs_rest,
    save_confusion_matrix_png, save_matrix_csv)

from torch.cuda.amp import autocast, GradScaler

# ---- DataParallel helper ----
def _unwrap(m):
    """Return underlying module when wrapped by nn.DataParallel; otherwise return as-is."""
    return m.module if isinstance(m, nn.DataParallel) else m


# 两个独立开关：训练期 AMP；评估期 AMP（默认评估不开，确保可比性）
AMP_TRAIN = False
AMP_EVAL  = False


TESTSETS = ["original","Ishihara","hor4","hor8","ver4","ver8"]
KEY_EVAL = ["Ishihara","hor4","hor8","ver4","ver8"]


def _should_save_epoch(epoch: int, every: int) -> bool:
    if every <= 0:
        return epoch == 0
    return (epoch % every) == 0

def _has_attr(obj, name: str) -> bool:
    return hasattr(obj, name)

def build_backbone(name: str, num_classes: int, pretrained: bool=True):
    """
    兼容老/新 torchvision：
    - 新API：weights=XXX_Weights.IMAGENET1K_*
    - 旧API：pretrained=True/False
    对于环境里没有的模型（如老版本无 convnext/efficientnet），自动回退或报清晰错误。
    """
    n = name.lower()

    def _resnet50():
        if _has_attr(models, "ResNet50_Weights"):
            wt = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = models.resnet50(weights=wt)
        else:
            m = models.resnet50(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m

    def _resnet101():
        if _has_attr(models, "ResNet101_Weights"):
            wt = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            m = models.resnet101(weights=wt)
        else:
            m = models.resnet101(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m

    def _inception_v3():
        if _has_attr(models, "Inception_V3_Weights"):
            wt = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.inception_v3(weights=wt, aux_logits=False)
        else:
            m = models.inception_v3(pretrained=pretrained, aux_logits=False)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m

    def _convnext_tiny():
        if not _has_attr(models, "convnext_tiny"):
            raise RuntimeError("convnext_tiny 在你当前 torchvision 版本里不可用；请用 --model_name resnet50 或升级 torchvision。")
        if _has_attr(models, "ConvNeXt_Tiny_Weights"):
            wt = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.convnext_tiny(weights=wt)
        else:
            m = models.convnext_tiny(pretrained=pretrained)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
        return m

    def _efficientnet_b4():
        if not _has_attr(models, "efficientnet_b4"):
            raise RuntimeError("efficientnet_b4 在你当前 torchvision 版本里不可用；请用 --model_name resnet50 或升级 torchvision。")
        if _has_attr(models, "EfficientNet_B4_Weights"):
            wt = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.efficientnet_b4(weights=wt)
        else:
            m = models.efficientnet_b4(pretrained=pretrained)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    # 路由
    if n in ["resnet50","r50"]:
        return _resnet50()
    elif n in ["resnet101","r101"]:
        return _resnet101()
    elif n in ["inception_v3","inception"]:
        return _inception_v3()
    elif n in ["convnext_t","convnext_tiny","convnext"]:
        return _convnext_tiny()
    elif n in ["efficientnet_b4","effb4","eb4"]:
        return _efficientnet_b4()
    else:
        # 默认回退到 resnet18，也做新旧API兼容
        if _has_attr(models, "ResNet18_Weights"):
            wt = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.resnet18(weights=wt)
        else:
            m = models.resnet18(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m


def _apply_augmix_then_norm(x_raw, model, targets, criterion, jsd_weight,da_module=None, da_eps_min=None, da_eps_max=None,return_fed=False,
                            use_ckpt=False):
    """
    x_raw: [0,1] 空间的图像张量（已做 28->224, AG 等像素域处理）
    流程：AugMix(aug1,aug2) -> Normalize -> (DA on aug1/aug2) -> forward
         clean:   Normalize -> (no DA) -> forward
    """
    # 1) 像素域 AugMix
    x_aug1 = augmix_batch(x_raw.clone())
    x_aug2 = augmix_batch(x_raw.clone())

    # 2) 归一化
    x_clean = normalize_imagenet(x_raw)
    x_aug1 = normalize_imagenet(x_aug1)
    x_aug2 = normalize_imagenet(x_aug2)

    # 3) 仅在 AugMix 分支上加 DeepAugment（Noise2Net-DA）
    if da_module is not None:
        _emin = 0.375 if da_eps_min is None else da_eps_min
        _emax = 0.75 if da_eps_max is None else da_eps_max
        x_aug1 = da_module(x_aug1, eps_min=_emin, eps_max=_emax)
        x_aug2 = da_module(x_aug2, eps_min=_emin, eps_max=_emax)
        # 注意：clean 不加 DA，保持 CE 的“干净监督”

    # 4) 前向 & 损失
    def _run(x):
        return model(x)

    if use_ckpt:
        x_clean = x_clean.detach().requires_grad_(True)
        x_aug1 = x_aug1.detach().requires_grad_(True)
        x_aug2 = x_aug2.detach().requires_grad_(True)
        logits_clean = checkpoint(_run, x_clean)
        logits_aug1 = checkpoint(_run, x_aug1)
        logits_aug2 = checkpoint(_run, x_aug2)
    else:
        logits_clean = model(x_clean)
        logits_aug1 = model(x_aug1)
        logits_aug2 = model(x_aug2)

    loss_ce  = criterion(logits_clean, targets)
    loss_jsd = jsd_weight * jsd_loss(logits_clean, logits_aug1, logits_aug2)
    loss = loss_ce + loss_jsd

    if return_fed:
        # 可选：返回“最终送入模型”的三路输入，方便你保存核验
        return loss, dict(clean=x_clean.detach(), aug1=x_aug1.detach(), aug2=x_aug2.detach())
    return loss

def evaluate(model, loader, device, name, epoch, save_pic: bool):
    model.eval(); correct=0; total=0; logits_all=[]; labels_all=[]
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with autocast(enabled=AMP_EVAL):
                logits = model(x)
            if save_pic and i==2:
                save_image(x[2], f"saved_samples/{name}_ep{epoch}_batch3_img3.png")
                print(f"[评估:{name}] 第3批第3张图已保存，mean={x.mean():.4f}, std={x.std():.4f}")
            preds = logits.argmax(1); total += y.size(0); correct += (preds==y).sum().item()
            logits_all.append(logits.cpu()); labels_all.append(y.cpu())
    logits=torch.cat(logits_all); labels=torch.cat(labels_all)
    acc = correct/total; nll = negative_log_likelihood(logits, labels).item()
    ece = expected_calibration_error(logits, labels, n_bins=15)
    xs, ys = reliability_points(logits, labels, n_bins=15)
    return acc, nll, ece, xs, ys

def main():
    parser = argparse.ArgumentParser()
    # 尽量兼容你原先参数名
    parser.add_argument('-model_name', '--model_name', default='resnet50', help='骨干网络名称')
    parser.add_argument('-device', '--device', default='cuda:0')
    parser.add_argument('-epochs', '--epochs', type=int, default=100)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=128)
    parser.add_argument('-num_workers', '--num_workers', type=int, default=8)
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('--pretrained', action='store_true', help='是否加载ImageNet预训练权重')
    parser.add_argument('--opt', choices=['adamw','sgd'], default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据与采样
    parser.add_argument('--mnist_root', default='./datasets')
    parser.add_argument('--ish_root',   default='./datasets/png224')
    parser.add_argument('--cap_train', type=int, default=1000, help='Ishihara每类上限（论文对齐）')
    parser.add_argument('--cap_test',  type=int, default=200)
    parser.add_argument('--download', action='store_true', help='若datasets目录缺失MNIST则下载')
    # 验证与选择
    parser.add_argument('--use_val_selection', action='store_true', default=True, help='用验证集选择best_epoch')
    parser.add_argument('--val_cap_per_class', type=int, default=100, help='Ishihara每类验证样本数')
    parser.add_argument('--min_delta', type=float, default=0.0, help='best提升阈值（防抖）')
    parser.add_argument('--test_each_epoch', action='store_true', default=True, help='每个epoch都在6个测试集上评测（仅记录，不用于选择）')
    parser.add_argument('--final_test_only', action='store_true', default=False, help='若开则训练期不跑测试集，最后一次用best.ckpt跑测试')
    parser.add_argument('--compute_auroc', action='store_true', help='最终评测时计算AUROC(OVR macro)')
    parser.add_argument('--final_results_file', default='final_results.csv', help='最终评测CSV文件名（默认 final_results.csv）')

    # 策略与增强
    parser.add_argument('--strategy', choices=['S0','S1','S2'], default='S2')
    parser.add_argument('--augmix', action='store_true')
    parser.add_argument('--augmix_jsd', type=float, default=12.0)
    parser.add_argument('--deepaugment', action='store_true')
    parser.add_argument('--da_blocks', type=int, default=2)
    parser.add_argument('--da_eps_min', type=float, default=0.375)
    parser.add_argument('--da_eps_max', type=float, default=0.75)
    parser.add_argument('--amp_train', action='store_true', default=True,
                        help='训练期启用混合精度（S1/S2 生效，S0 固定关闭）')
    parser.add_argument('--amp_eval', action='store_true', default=False,
                        help='评估期启用混合精度（默认关闭以保证各策略可比）')
    parser.add_argument('--use_ckpt', action='store_true',
                        help='在 S2 的三路前向上启用 gradient checkpointing（降显存，略慢）')
    # 在现有 parser.add_argument 们后面加一行
    parser.add_argument('--save_eval_imgs_every', type=int, default=1,
                        help='0=只在第0个epoch保存；N=每隔N个epoch保存一次（含第0个）')

    # 结果
    parser.add_argument('--results_file', default='results.csv')
    parser.add_argument('--dp', action='store_true', help='Use nn.DataParallel across all available GPUs on this machine')

    args = parser.parse_args()
    print(f"🎲 seed={args.seed}")
    global AMP_TRAIN, AMP_EVAL
    AMP_TRAIN = bool(args.amp_train and args.strategy in ['S1', 'S2'] and torch.cuda.is_available())
    AMP_EVAL = bool(args.amp_eval and torch.cuda.is_available())
    scaler = GradScaler(enabled=AMP_TRAIN)

    if args.strategy == 'S0':
        args.augmix = False; args.deepaugment = False
    elif args.strategy == 'S1':
        args.augmix = False; args.deepaugment = False
    elif args.strategy == 'S2':
        args.augmix = True; args.deepaugment = True

    # 固定随机种子
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # 模型与优化器
    model = build_backbone(args.model_name, num_classes=10, pretrained=args.pretrained).to(device)
    # ★ Enable DataParallel across available GPUs on this machine if requested
    if args.dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        print(f"🧩 启用 DataParallel：{torch.cuda.device_count()} x GPU，device_ids={device_ids}，主卡=cuda:0")
        model = nn.DataParallel(model, device_ids=device_ids).to(device)

    print("✅ 模型初始化完成：", args.model_name)
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) if args.opt=='adamw' \
        else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    # === Ishihara 训练/验证严格不重叠 ===
    ish_train, ish_val = build_ishihara_train_val(
        root=args.ish_root, plates=[2,3,5,6,7,9],
        cap_train=args.cap_train, cap_val=args.val_cap_per_class, seed=args.seed
    )
    ish_loader = DataLoader(ish_train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"✅ Ishihara 训练集就绪（每类{args.cap_train}），验证集就绪（每类{args.val_cap_per_class}）。")
    mnist_train = mnist_train_dataset(root=args.mnist_root, download=args.download)

    # 测试集（固定：ori224、Ish p{2,4,8}、AG四套）
    print("⌛ 正在构建 6 个测试加载器...")
    test_loaders = {}
    test_loaders["original"] = DataLoader(
        mnist_test_dataset(args.mnist_root, download=args.download),
        batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_and_to224(b)
    )
    ish_test = build_ishihara_png(args.ish_root, split='test', plates=[2,4,8],
                                  cap_per_class=args.cap_test, seed=args.seed)
    test_loaders["Ishihara"] = DataLoader(ish_test, batch_size=64, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True,
                                          collate_fn=lambda b: collate_and_to224(b))
    from functools import partial
    def collate_mnist_ag(batch, interval:int, direction:str):
        imgs, labels = zip(*batch)
        x = torch.stack(imgs, dim=0)  # [B,1,28,28]
        dir_vec = (1,0) if direction=='hor' else (0,1)
        x = ag_distort_224(x, threshold=0.5, interval=interval, phase=interval//2, direction=dir_vec)
        x = normalize_imagenet(x.float())
        y = torch.tensor(labels, dtype=torch.long); return x, y

    for name, spec in {
        "hor4": dict(interval=4, direction='hor'),
        "hor8": dict(interval=8, direction='hor'),
        "ver4": dict(interval=4, direction='ver'),
        "ver8": dict(interval=8, direction='ver'),
    }.items():
        test_loaders[name] = DataLoader(
            mnist_test_dataset(args.mnist_root, download=args.download),
            batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True,
            collate_fn=partial(collate_mnist_ag, interval=spec['interval'], direction=spec['direction'])
        )
    print("✅ 测试加载器就绪。")

    # === 验证：AG6（hor/ver）作为代理 OOD（避免与测试 4/8 重叠）===
    ag6_hor_val_loader = DataLoader(
        mnist_test_dataset(args.mnist_root, download=args.download),
        batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        collate_fn=partial(collate_mnist_ag, interval=6, direction='hor')
    )
    ag6_ver_val_loader = DataLoader(
        mnist_test_dataset(args.mnist_root, download=args.download),
        batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        collate_fn=partial(collate_mnist_ag, interval=6, direction='ver')
    )

    def _eval_loader_acc(model, loader, device, name, epoch,
                     save_val_sample=False):
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(enabled=AMP_EVAL):
                    logits = model(x)
                preds = logits.argmax(1)
                total += y.size(0)
                correct += (preds == y).sum().item()
                if save_val_sample and i == 2:
                    save_image(x[2], f"saved_samples/{name}_ep{epoch}_batch3_img3_input.png")
                if (i + 1) % 50 == 0:
                    bs = x.size(0)
                    print(f"[评估:{name}] 已处理 {(i + 1) * bs} 张")
        acc = correct / total if total > 0 else 0.0
        print(f" 验证集[Val] {name}: acc={acc:.4f}")
        return acc

    # 增强模块
    da_module = Noise2NetDA(blocks=args.da_blocks).to(device) if args.deepaugment else None
    os.makedirs("saved_samples", exist_ok=True)

    # 训练循环
    results_rows = []
    best_metric = -1e9
    best_epoch_sel = -1
    best_subset_mean = -1.0; best_epoch = -1
    print("🚀 开始训练！策略:", args.strategy)
    print(
        f"[cfg] model={args.model_name}, strategy={args.strategy},seed={args.seed}, "
        f"augmix={args.augmix}, deepaugment={args.deepaugment}, "
        f"use_ckpt={getattr(args, 'use_ckpt', False)}, "
        f"amp_train={AMP_TRAIN}, amp_eval={AMP_EVAL}, "
        f"da_eps=[{args.da_eps_min:.3f},{args.da_eps_max:.3f}], "
        f"batch_size={args.batch_size}, device={args.device}"
    )

    for epoch in range(args.epochs):
        SAVE_THIS_EPOCH = _should_save_epoch(epoch, args.save_eval_imgs_every)
        # 本轮计数器
        train_seen_epoch = 0
        # test_seen_epoch / val_seen_epoch 在各自阶段里统计

        model.train(); running_loss=0.0; total_updates=0
        t0 = time.time()
        mnist_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

        def sample_ag(x28):
            interval = random.choice([4,8]); direction = random.choice(['hor','ver'])
            dir_vec = (1,0) if direction=='hor' else (0,1)
            return ag_distort_224(x28, threshold=0.5, interval=interval, phase=interval//2, direction=dir_vec)

        if args.strategy == 'S0':
            assert (not args.augmix) and (not args.deepaugment), "S0 不能开启 AugMix / DeepAugment"
            for i,(xb_i,yb_i) in enumerate(ish_loader):
                xi_raw = xb_i.to(device, non_blocking=True).float()
                yi = yb_i.to(device, non_blocking=True)
                train_seen_epoch += yi.size(0)
                if i==0 and epoch==0:
                    save_image(xi_raw[0], f"saved_samples/未归一化ep{epoch}_ish_clean.png")
                    print("🖼️ 已保存 Ishihara 样例图：saved_samples/未归一化ep0_ish_clean.png")
                if args.augmix:
                    loss_i, fed_i = _apply_augmix_then_norm(
                        xi_raw, model, yi, criterion, args.augmix_jsd,
                        da_module=da_module, da_eps_min=args.da_eps_min, da_eps_max=args.da_eps_max,return_fed=True
                    )
                    if i == 0 and epoch == 0:
                        save_image(fed_i["clean"][0], f"saved_samples/train_ep{epoch}_ish_input.png")
                else:
                    xi = normalize_imagenet(xi_raw)
                    if da_module is not None:
                        xi = da_module(xi)
                    if i == 0 and epoch == 0:
                        save_image(xi[0].detach().cpu(), f"saved_samples/train_ep{epoch}_ish_input.png")
                        print(f"[dump] s0策略：saved_samples/train_ep{epoch}_ish_input.png")
                    logits = model(xi)
                    loss_i = criterion(logits, yi)

                optimizer.zero_grad(set_to_none=True); loss_i.backward(); optimizer.step()
                running_loss += float(loss_i.item()); total_updates += 1
                if (i + 1) % 50 == 0:
                    seen = (i + 1) * args.batch_size
                    print(f"[训练:S0] epoch={epoch} 已处理 {seen} 张 Ishihara 图像")
        else:
            for batch_idx, ((xb_i,yb_i),(xb_m,yb_m)) in enumerate(zip(ish_loader, mnist_loader)):
                # Ishihara
                xi_raw = xb_i.to(device, non_blocking=True).float()
                yi = yb_i.to(device, non_blocking=True)
                train_seen_epoch += yi.size(0)  # ← Ishihara 计数
                # Ishihara 支路
                if args.strategy == 'S2' and args.augmix:
                    with autocast(enabled=AMP_TRAIN):
                        out = _apply_augmix_then_norm(
                        xi_raw, model, yi, criterion, args.augmix_jsd,
                        da_module=da_module,da_eps_min=args.da_eps_min, da_eps_max=args.da_eps_max,
                        return_fed=(epoch == 0 and batch_idx == 0),use_ckpt=args.use_ckpt
                    )

                    if isinstance(out, tuple):
                        loss_i, fed = out
                        if epoch == 0 and batch_idx == 0:
                            save_image(fed["clean"][0], f"saved_samples/train_ep{epoch}_ish_clean_input.png")
                            save_image(fed["aug1"][0], f"saved_samples/train_ep{epoch}_ish_aug1_input.png")
                            save_image(fed["aug2"][0], f"saved_samples/train_ep{epoch}_ish_aug2_input.png")
                    else:
                        loss_i = out
                else:
                    # S0/S1 走这里：只有“单路最终输入”
                    xi = normalize_imagenet(xi_raw)
                    if da_module is not None:
                        xi = da_module(xi)  # 注：在 S0/S1 中 da_module 本来就是 None
                    if batch_idx == 0 and epoch == 0:
                        save_image(xi[0].detach().cpu(), f"saved_samples/train_ep{epoch}_ish_input.png")
                        print(f"[dump] s1策略,saved_samples/train_ep{epoch}_ish_input.png")
                    with autocast(enabled=AMP_TRAIN):
                        logits = model(xi)
                        loss_i = criterion(logits, yi)

                # ---------------- [II] 先回传 Ish（释放这条大图） ----------------
                optimizer.zero_grad(set_to_none=True)
                loss_i_scalar = float(loss_i.item())  # 仅日志用
                if AMP_TRAIN:
                    scaler.scale(0.5 * loss_i).backward()
                else:
                    (0.5 * loss_i).backward()
                del loss_i  # 释放 Ish 的计算图，降低峰值显存

                # MNIST-AG
                xm28 = xb_m.to(device, non_blocking=True).float()
                ym = yb_m.to(device, non_blocking=True)
                train_seen_epoch += ym.size(0)  # ← MNIST-AG 计数
                xm_raw = sample_ag(xm28)
                if epoch==0 and total_updates==0:
                    save_image(xm_raw[0], f"saved_samples/未归一化ep{epoch}_mnist_ag.png")
                    print("🖼️ 已保存 MNIST-AG 样例图：saved_samples/未归一化ep0_mnist_ag.png")

                if args.strategy == 'S2' and args.augmix:
                    with autocast(enabled=AMP_TRAIN):
                        out_m = _apply_augmix_then_norm(
                            xm_raw, model, ym, criterion, args.augmix_jsd,
                            da_module=da_module, da_eps_min=args.da_eps_min, da_eps_max=args.da_eps_max,
                            return_fed=(epoch == 0 and batch_idx == 0),use_ckpt=args.use_ckpt
                        )
                    if isinstance(out_m, tuple):
                        loss_m, fed_m = out_m
                        if epoch == 0 and batch_idx == 0:
                            save_image(fed_m["clean"][0], f"saved_samples/train_ep{epoch}_mnist_ag_clean_input.png")
                            save_image(fed_m["aug1"][0], f"saved_samples/train_ep{epoch}_mnist_ag_aug1_input.png")
                            save_image(fed_m["aug2"][0], f"saved_samples/train_ep{epoch}_mnist_ag_aug2_input.png")
                    else:
                        loss_m = out_m
                else:
                    xm = normalize_imagenet(xm_raw)
                    if da_module is not None:
                        xm = da_module(xm)
                    if epoch == 0 and batch_idx == 0:
                        save_image(xm[0].detach().cpu(), f"saved_samples/train_ep{epoch}_mnist_ag_input.png")
                        print(f"[dump] saved_samples/train_ep{epoch}_mnist_ag_input.png")
                    with autocast(enabled=AMP_TRAIN):
                        logits_m = model(xm)
                        loss_m = criterion(logits_m, ym)

                # ---------------- [IV] 回传 MNIST-AG 并 step ----------------
                loss_m_scalar = float(loss_m.item())
                if AMP_TRAIN:
                    scaler.scale(0.5 * loss_m).backward()
                    scaler.step(optimizer);
                    scaler.update()
                else:
                    (0.5 * loss_m).backward()
                    optimizer.step()
                del loss_m

                # ------- 日志累计 -------
                loss_scalar = 0.5 * (loss_i_scalar + loss_m_scalar)
                running_loss += loss_scalar
                total_updates += 1


                if (batch_idx + 1) % 50 == 0:
                    print(f"[训练:{args.strategy}] epoch={epoch} 已处理 {train_seen_epoch} 张图像（Ish + MNIST-AG）")
        print(f"🧮 [训练:{args.strategy}] epoch={epoch} 本轮训练共使用 {train_seen_epoch} 张图像")


        # 评估（若 final_test_only=False 则每轮都测 6 个测试集，仅记录曲线）
        if not args.final_test_only:
            test_seen_epoch = 0  # ← 新增
            subset_accs=[]; epoch_metrics={}
            for name, loader in test_loaders.items():
                acc,nll,ece,xs,ys = evaluate(model, loader, device, name, epoch, save_pic=SAVE_THIS_EPOCH)
                epoch_metrics[name] = dict(acc=acc,nll=nll,ece=ece,xs=xs,ys=ys)
                results_rows.append(dict(epoch=epoch, testset=name, acc=acc, nll=nll, ece=ece,
                                         backbone=args.model_name, strategy=args.strategy, seed=args.seed))
                # ← 新增：按数据集大小计数（以原始样本计）
                test_seen_epoch += len(loader.dataset)
            for key in KEY_EVAL: subset_accs.append(epoch_metrics[key]['acc'])
            subset_mean = sum(subset_accs)/len(subset_accs) if subset_accs else 0.0
            dt = time.time()-t0
            ood_text = ", ".join([f"{k}={epoch_metrics[k]['acc']:.4f}" for k in KEY_EVAL])
            print(f"📊 [Epoch {epoch}] OOD各集 acc: {ood_text}")
            print(f"📣 [Epoch {epoch}] 步数≈{total_updates}, loss={running_loss:.3f}, OOD五集均值={subset_mean:.4f}, 用时={dt:.1f}s")
            # ← 新增：本轮测试总量
            print(f"🧪 [测试] epoch={epoch} 本轮测试共使用 {test_seen_epoch} 张图像（6 个测试集总和）")
            # 按 OOD(5) 均值保存一个“最好 OOD epoch”的快照（仅记录，不影响验证选优）
            if subset_mean > best_subset_mean:
                best_subset_mean = subset_mean; best_epoch = epoch
                os.makedirs(f"best_ood_epoch_{epoch}", exist_ok=True)
                torch.save(_unwrap(model).state_dict(), f"best_ood_epoch_{epoch}/best_weights.pth")
                main._ood_best_epoch = epoch
                main._ood_best_mean = subset_mean

        # ---- 验证选择：Ish_val + AG6(hor/ver) ----
        if args.use_val_selection:
            # Ish_val
            acc_ish_val = _eval_loader_acc(model,
                DataLoader(ish_val, batch_size=64, shuffle=False, num_workers=args.num_workers,
                           pin_memory=True, collate_fn=lambda b: collate_and_to224(b)),
                device, "Ish_val", epoch, save_val_sample=SAVE_THIS_EPOCH)

            # AG6 val（hor / ver）
            acc_ag6_h = _eval_loader_acc(model, ag6_hor_val_loader, device, "AG6_hor_val", epoch, save_val_sample=SAVE_THIS_EPOCH)
            acc_ag6_v = _eval_loader_acc(model, ag6_ver_val_loader, device, "AG6_ver_val", epoch, save_val_sample=SAVE_THIS_EPOCH)
            mean_acc_val = (acc_ish_val + acc_ag6_h + acc_ag6_v) / 3.0
            val_imgs_seen_epoch = len(ish_val) \
                                  + len(ag6_hor_val_loader.dataset) \
                                  + len(ag6_ver_val_loader.dataset)
            print(f"🔍 [验证] epoch={epoch} 本轮验证共使用 {val_imgs_seen_epoch} 张图像（Ish_val + AG6_hor + AG6_ver）")

            print(f"📏 [Val] mean_acc_val={mean_acc_val:.4f}")
            if mean_acc_val > getattr(main, "_best_metric", -1e9) + args.min_delta:
                main._best_metric = mean_acc_val
                main._best_epoch_sel = epoch
                torch.save(_unwrap(model).state_dict(), './best.ckpt')
                print(f"💾 验证集提升，已保存 best.ckpt（epoch={epoch}）")
        else:
            main._best_metric = float("nan")
            main._best_epoch_sel = epoch
            torch.save(_unwrap(model).state_dict(), './best.ckpt')

    # 写 CSV（每轮测试日志）
    import csv
    with open(args.results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","testset","acc","nll","ece","backbone","strategy","seed"])
        writer.writeheader(); writer.writerows(results_rows)
    print(f"📄 指标已写入 {args.results_file}")

    # ===== Final test with best.ckpt =====
    try:
        state = torch.load('./best.ckpt', map_location=device)
        _unwrap(model).load_state_dict(state)
        print(f"✅ 已加载 best.ckpt（来自 epoch={getattr(main,'_best_epoch_sel','?')}，val_metric={getattr(main,'_best_metric','?')}）")
    except Exception as e:
        print("⚠️ 加载 best.ckpt 失败，改用当前权重进行最终评测：", e)

    # 最终一次：全量六个测试集，输出扩展指标与混淆矩阵
    out_final = args.final_results_file
    with open(out_final, "w", newline="") as f:
        header = ["ood_epoch","OOD_best_mean","val_epoch","final_best_mean",
                  "testset","acc","nll","ece","macro_precision","macro_recall","macro_f1",
                  "micro_precision","micro_recall","micro_f1"]
        if args.compute_auroc: header.append("auroc_ovr_macro")
        writer = csv.DictWriter(f, fieldnames=header); writer.writeheader()
        os.makedirs("final_cm", exist_ok=True)  # ← 放在写 CSV/CM 之前

        final_rows = []
        ood_names = ["Ishihara", "hor4", "hor8", "ver4", "ver8"]
        ood_accs_tmp = {}

        for name, loader in test_loaders.items():
            model.eval(); all_logits=[]; all_labels=[]
            with torch.no_grad():
                for x,y in loader:
                    x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                    logits=model(x); all_logits.append(logits.cpu()); all_labels.append(y.cpu())
            logits = torch.cat(all_logits); labels = torch.cat(all_labels)
            acc = float((logits.argmax(1)==labels).float().mean().item())
            nll = negative_log_likelihood(logits, labels).item()
            ece = expected_calibration_error(logits, labels, n_bins=15)
            cm = confusion_matrix_from_logits(logits, labels, num_classes=10)
            prf = prf1_from_confusion_matrix(cm)
            row = dict(testset=name, acc=acc, nll=nll, ece=ece,
                       macro_precision=prf["macro_precision"], macro_recall=prf["macro_recall"], macro_f1=prf["macro_f1"],
                       micro_precision=prf["micro_precision"], micro_recall=prf["micro_recall"], micro_f1=prf["micro_f1"])
            if args.compute_auroc:
                auc = auroc_one_vs_rest(logits, labels)
                row["auroc_ovr_macro"] = auc if auc is not None else ""

            save_confusion_matrix_png(cm, f"final_cm/{name}_cm.png")
            save_matrix_csv(cm, f"final_cm/{name}_cm.csv")
            # 收集 OOD5 acc
            if name in ood_names:
                ood_accs_tmp[name] = acc

            final_rows.append(row)

        # 计算最终（best.ckpt 下）OOD5 五集的均值
        final_best_mean = (sum(ood_accs_tmp.values()) / len(ood_accs_tmp)) if ood_accs_tmp else float("nan")

        # 取出两个 epoch 指标：训练期 OOD-best 与 验证选优的 best.ckpt epoch
        ood_epoch = getattr(main, "_ood_best_epoch", "")
        ood_best_mean = getattr(main, "_ood_best_mean", "")
        val_epoch = getattr(main, "_best_epoch_sel", "")

        # 第二遍：把四个前置字段填进去，再写每一行
        for row in final_rows:
            row_out = dict(
                ood_epoch=ood_epoch,
                OOD_best_mean=ood_best_mean,
                val_epoch=val_epoch,
                final_best_mean=final_best_mean,
            )
            row_out.update(row)
            writer.writerow(row_out)

    print(f"📄 已输出最终结果到 {out_final}，并保存混淆矩阵到 final_cm/")

if __name__ == "__main__":
    main()