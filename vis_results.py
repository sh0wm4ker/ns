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

# ================= 中文支持设置 =================
# 设置中文字体，防止中文显示乱码
# Windows 常用: 'SimHei', 'Microsoft YaHei'
# Linux/Mac 常用: 'Arial Unicode MS', 'PingFang SC'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# ===============================================

# 确保能导入 Net
try:
    from model.net import Net
except ImportError:
    print("Error: Could not import 'Net' from 'model.net'.")
    exit()

# ================= 配置 =================
# 路径保持不变
IMG_PATH = r"val_pic\agricultural21.tif"
CKPT_PATH = r"checkpoint_ucm/0279.ckpt"
SAVE_NAME = "epoch478_vis_adaptive_chinese.png"

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
    """执行一次标准推理"""
    with torch.no_grad():
        # 1. 编码
        z3 = net.a_model(x_input)
        z3_syntax = z3[:, :net.M, :, :]
        z3_content = z3[:, net.M:, :, :]

        # 2. 应用 syntax_model 进行降维
        z3_syntax = net.syntax_model(z3_syntax)

        # 3. 量化
        z3_syntax_hat = torch.round(z3_syntax)
        z3_content_hat = torch.round(z3_content)

        # 4. 生成权重 & 解码
        conv_weights = net.conv_weights_gen(z3_syntax_hat)

        x_tilde = net.s_model(z3_content_hat)
        x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

        # 5. 后处理 (如果模型是 Stage2 训练的)
        if net.post_processing:
            x_tilde_bf = net.han_head(x_tilde_bf)

        # 6. 反归一化
        x_recon = torch.clamp((x_tilde_bf + 1) / 2.0, 0, 1)
    return x_recon


def run_adaptive_finetune(net, x_input):
    """在线适应 (Online Adaptation)"""
    print(f"正在进行在线自适应微调 ({TUNE_ITER} 次迭代)...")

    # 1. 冻结除编码器以外的所有参数
    for param in net.parameters():
        param.requires_grad = False
    for param in net.a_model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(net.a_model.parameters(), lr=LR)

    net.train()

    for i in range(TUNE_ITER):
        optimizer.zero_grad()

        # 前向传播
        z3 = net.a_model(x_input)
        z3_syntax = z3[:, :net.M, :, :]
        z3_content = z3[:, net.M:, :, :]

        # 训练时也必须经过 syntax_model
        z3_syntax = net.syntax_model(z3_syntax)

        # 传入未量化的特征以保留梯度
        conv_weights = net.conv_weights_gen(z3_syntax)

        x_tilde = net.s_model(z3_content)
        x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

        # 简化版重建 (只优化MSE)
        x_recon_train = (x_tilde_bf + 1) / 2.0

        target = (x_input + 1) / 2.0
        mse_loss = torch.mean((x_recon_train - target) ** 2)

        loss = mse_loss * 255 * 255

        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"  Iter {i + 1}/{TUNE_ITER}, Loss: {loss.item():.4f}")

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
    net = Net((1, 256, 256, 3), (1, 256, 256, 3), is_high=False, post_processing=False).cuda()

    try:
        net.load_state_dict(torch.load(CKPT_PATH, map_location='cuda'))
        print("权重加载成功。")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    # 3. [阶段 A] 标准推理
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
    axes[0].set_title("原始遥感影像\n(Ground Truth)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    rect0 = patches.Rectangle((CROP_BOX[0], CROP_BOX[1]), CROP_BOX[2], CROP_BOX[3], linewidth=2, edgecolor='red',
                              facecolor='none')
    axes[0].add_patch(rect0)

    # (B) Standard
    axes[1].imshow(img_std)
    axes[1].set_title(f"基础重建结果\nPSNR: {psnr_std:.2f} dB", fontsize=14)
    axes[1].axis('off')
    rect1 = patches.Rectangle((CROP_BOX[0], CROP_BOX[1]), CROP_BOX[2], CROP_BOX[3], linewidth=2, edgecolor='red',
                              facecolor='none')
    axes[1].add_patch(rect1)

    # (C) Adaptive
    axes[2].imshow(img_adt)
    title_color = 'darkgreen' if psnr_adt > psnr_std else 'black'
    axes[2].set_title(f"自适应重建结果 (Ours)\nPSNR: {psnr_adt:.2f} dB", fontsize=14, color=title_color,
                      fontweight='bold')
    axes[2].axis('off')
    rect2 = patches.Rectangle((CROP_BOX[0], CROP_BOX[1]), CROP_BOX[2], CROP_BOX[3], linewidth=2, edgecolor='red',
                              facecolor='none')
    axes[2].add_patch(rect2)

    # (D) Error Map
    im = axes[3].imshow(diff_norm, cmap='jet', vmin=0, vmax=0.5)
    axes[3].set_title("重建误差热力图\n(Error Map)", fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300, bbox_inches='tight')
    print(f"\n成功生成对比图: {SAVE_NAME}")
    print(f"标准 PSNR: {psnr_std:.2f} dB")
    print(f"适应后 PSNR: {psnr_adt:.2f} dB")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    generate_vis()