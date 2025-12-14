import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import math
import copy

# 确保能导入 Net
try:
    from model.net import Net
except ImportError:
    print("Error: Could not import 'Net' from 'model.net'.")
    exit()

# ================= 配置 =================
IMG_PATH = r"val_pic\agricultural00.tif"
CKPT_PATH = r"checkpoint_ucm/0099.ckpt"
SAVE_NAME = "vis_adaptive_comparison.png"

# 微调参数
LAMBDA = 0.01
TUNE_ITER = 60
LR = 1e-3

# 局部放大区域 (x, y, w, h)
CROP_BOX = (50, 50, 150, 150)


# =======================================

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(1. / mse)


def run_inference(net, x_input):
    """执行一次标准推理 (严格遵循 model.net 的 forward 逻辑)"""
    with torch.no_grad():
        # 1. 编码
        z3 = net.a_model(x_input)
        z3_syntax = z3[:, :net.M, :, :]
        z3_content = z3[:, net.M:, :, :]

        # 2. 量化 (Inference 时必须量化)
        z3_syntax_hat = torch.round(z3_syntax)
        z3_content_hat = torch.round(z3_content)

        # 3. 生成权重 & 解码
        # 【修正】直接传入 4D 张量，不要做 torch.mean
        conv_weights = net.conv_weights_gen(z3_syntax_hat)

        x_tilde = net.s_model(z3_content_hat)
        x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

        # 4. 后处理 (Post-Processing)
        # 如果训练了 HAN (Stage 2)，这里应该开启。Stage 1 关闭。
        if net.post_processing:
            x_tilde_bf = net.han_head(x_tilde_bf)

        # 5. 反归一化
        x_recon = torch.clamp((x_tilde_bf + 1) / 2.0, 0, 1)
    return x_recon


def run_adaptive_finetune(net, x_input):
    """在线适应 (参考 eval.py 的 pre-processing 逻辑)"""
    print(f"正在进行在线自适应微调 ({TUNE_ITER} 次迭代)...")

    # 1. 冻结除编码器以外的所有参数
    for param in net.parameters():
        param.requires_grad = False
    for param in net.a_model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(net.a_model.parameters(), lr=LR)

    # 临时开启训练模式 (为了梯度传播)
    net.train()

    for i in range(TUNE_ITER):
        optimizer.zero_grad()

        # 前向传播 (不使用 round，保留梯度)
        z3 = net.a_model(x_input)
        z3_syntax = z3[:, :net.M, :, :]
        z3_content = z3[:, net.M:, :, :]

        # 【修正】直接传入 4D 张量
        # 注意：这里传入未量化的 z3_syntax 以便梯度回传到 a_model
        conv_weights = net.conv_weights_gen(z3_syntax)

        x_tilde = net.s_model(z3_content)
        x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

        # 简化版重建 (跳过 post_processing 以节省显存/时间，且主要优化的是编码器特征)
        x_recon_train = (x_tilde_bf + 1) / 2.0

        # 计算 Loss (仅优化失真 MSE)
        target = (x_input + 1) / 2.0
        mse_loss = torch.mean((x_recon_train - target) ** 2)

        # 放大 Loss 避免梯度过小
        loss = mse_loss * 255 * 255

        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"  Iter {i + 1}/{TUNE_ITER}, Loss: {loss.item():.4f}")

    # 微调结束，切回 eval 模式进行最终推理
    net.eval()
    return run_inference(net, x_input)


def generate_vis():
    # 1. 准备数据
    if not os.path.exists(IMG_PATH):
        print(f"Error: 找不到图片 {IMG_PATH}")
        return

    img_pil = Image.open(IMG_PATH).convert('RGB')
    x = T.ToTensor()(img_pil).unsqueeze(0).cuda()
    x_input = x * 2.0 - 1.0

    # 2. 加载模型
    # 注意：这里 post_processing设为False，因为你的checkpoint可能是Stage1的
    # 如果是Stage2模型，请改为True
    net = Net((1, 256, 256, 3), (1, 256, 256, 3), is_high=False, post_processing=False).cuda()

    try:
        net.load_state_dict(torch.load(CKPT_PATH, map_location='cuda'))
        print("权重加载成功。")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    # 3. [阶段 A] 标准推理
    # 备份模型，防止微调影响对比
    net_standard = copy.deepcopy(net)
    x_recon_std = run_inference(net_standard, x_input)

    # 4. [阶段 B] 在线适应推理
    x_recon_adapt = run_adaptive_finetune(net, x_input)

    # 5. 绘图准备
    img_gt = x[0].permute(1, 2, 0).cpu().numpy()
    img_std = x_recon_std[0].permute(1, 2, 0).cpu().numpy()
    img_adt = x_recon_adapt[0].permute(1, 2, 0).cpu().numpy()

    psnr_std = calculate_psnr(img_gt, img_std)
    psnr_adt = calculate_psnr(img_gt, img_adt)

    diff = np.mean(np.abs(img_gt - img_adt), axis=2)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    # 6. 绘图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # (A) GT
    axes[0].imshow(img_gt)
    axes[0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    rect0 = patches.Rectangle((CROP_BOX[0], CROP_BOX[1]), CROP_BOX[2], CROP_BOX[3], linewidth=2, edgecolor='red',
                              facecolor='none')
    axes[0].add_patch(rect0)

    # (B) Standard
    axes[1].imshow(img_std)
    axes[1].set_title(f"Standard Recon\nPSNR: {psnr_std:.2f} dB", fontsize=14)
    axes[1].axis('off')
    rect1 = patches.Rectangle((CROP_BOX[0], CROP_BOX[1]), CROP_BOX[2], CROP_BOX[3], linewidth=2, edgecolor='red',
                              facecolor='none')
    axes[1].add_patch(rect1)

    # (C) Adaptive
    axes[2].imshow(img_adt)
    title_color = 'darkgreen' if psnr_adt > psnr_std else 'black'
    axes[2].set_title(f"Adaptive Recon (Ours)\nPSNR: {psnr_adt:.2f} dB", fontsize=14, color=title_color,
                      fontweight='bold')
    axes[2].axis('off')
    rect2 = patches.Rectangle((CROP_BOX[0], CROP_BOX[1]), CROP_BOX[2], CROP_BOX[3], linewidth=2, edgecolor='red',
                              facecolor='none')
    axes[2].add_patch(rect2)

    # (D) Error Map
    im = axes[3].imshow(diff_norm, cmap='jet', vmin=0, vmax=0.5)
    axes[3].set_title("Error Map (Adaptive)", fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300, bbox_inches='tight')
    print(f"\n成功生成对比图: {SAVE_NAME}")
    print(f"标准 PSNR: {psnr_std:.2f} dB")
    print(f"适应后 PSNR: {psnr_adt:.2f} dB")


if __name__ == "__main__":
    # 解决 OpenMP 冲突
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    generate_vis()