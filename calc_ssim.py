import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import math

# 引入你的模型
from model.net import Net


# ==========================================
# [核心] 内置 MS-SSIM 计算模块 (无需额外安装库)
# ==========================================
def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, K=(0.01, 0.03)):
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        return ssim_map.mean(), cs_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1), cs_map.mean(1).mean(1).mean(1)


def ms_ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, weights=None):
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []

    for i in range(levels):
        ssim_val, cs_val = _ssim(X, Y, win=win, data_range=data_range, size_average=True)
        mcs.append(cs_val)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    # 最后一个尺度的 SSIM 值作为最终的亮度对比项
    # MS-SSIM = Product(MCS[i] ^ weight[i]) * (SSIM_last ^ weight_last)
    # 这里为了简便，通常实现为:
    mcs = torch.stack(mcs)
    # 最后一个是 ssim_val
    mcs[-1] = ssim_val

    ms_ssim_val = torch.prod(mcs ** weights)
    return ms_ssim_val


# ==========================================
# 推理与计算主逻辑
# ==========================================
def calculate_metrics(data_path, weight_path, is_high, post_processing):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    # 注意：这里我们初始化一个临时的形状，后面加载权重即可
    net = Net((1, 256, 256, 3), (1, 256, 256, 3), is_high, post_processing).to(device)

    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        net.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print(f"Error: Weights not found at {weight_path}")
        return

    net.eval()

    total_ssim = 0.0
    total_ms_ssim = 0.0
    count = 0

    image_files = os.listdir(data_path)
    print(f"Found {len(image_files)} images in {data_path}")
    print("-" * 60)
    print(f"{'Image Name':<30} | {'MS-SSIM':<10} | {'SSIM':<10}")
    print("-" * 60)

    for img_name in image_files:
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue

        img_path = os.path.join(data_path, img_name)

        # 2. 图像预处理 (保持与 train/eval 一致)
        img_pil = Image.open(img_path).convert('RGB')
        w, h = img_pil.size

        # Padding logic from eval.py
        h_padded = h if h % 64 == 0 else (h // 64 + 1) * 64
        w_padded = w if w % 64 == 0 else (w // 64 + 1) * 64

        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Pad
        pad_h = h_padded - h
        pad_w = w_padded - w
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Normalize to [-1, 1] for network input
        input_tensor = img_tensor * 2.0 - 1.0

        # 3. 手动执行推理 (参考 vis_results.py 的逻辑)
        with torch.no_grad():
            # Encoder
            z3 = net.a_model(input_tensor)
            z3_syntax = z3[:, :net.M, :, :]
            z3_content = z3[:, net.M:, :, :]

            # Syntax Model & Quantization
            z3_syntax = net.syntax_model(z3_syntax)
            z3_syntax_hat = torch.round(z3_syntax)
            z3_content_hat = torch.round(z3_content)

            # Weight Gen & Decoder
            conv_weights = net.conv_weights_gen(z3_syntax_hat)
            x_tilde = net.s_model(z3_content_hat)
            x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

            # Post Processing
            if net.post_processing:
                x_tilde_bf = net.HAN(x_tilde_bf)  # 注意：这里原代码有可能是 net.han_head 或 net.HAN，请核对 net.py 定义
                # 根据你上传的 net.py，变量名是 self.HAN
                conv_weights_han = net.conv_weights_gen_HAN(z3_syntax_hat)
                x_tilde_bf = net.batch_conv(conv_weights_han, x_tilde_bf)
                x_tilde_bf = net.add_mean(x_tilde_bf)

            # 4. 转换回 0-255 范围进行评估
            # GT
            gt = torch.clamp((input_tensor + 1) * 127.5, 0, 255).round()
            # Recon
            recon = torch.clamp((x_tilde_bf + 1) * 127.5, 0, 255).round()

            # Crop back to original size (remove padding)
            gt = gt[:, :, :h, :w]
            recon = recon[:, :, :h, :w]

            # 5. 计算指标
            # MS-SSIM
            val_ms_ssim = ms_ssim(gt, recon, data_range=255.0).item()
            # Standard SSIM (using just one scale)
            # 我们复用 _ssim 函数，构建一个简单的 window
            win = _fspecial_gauss_1d(11, 1.5).repeat(3, 1, 1, 1)
            val_ssim, _ = _ssim(gt, recon, win=win, data_range=255.0)
            val_ssim = val_ssim.item()

            print(f"{img_name:<30} | {val_ms_ssim:.4f}     | {val_ssim:.4f}")

            total_ms_ssim += val_ms_ssim
            total_ssim += val_ssim
            count += 1

    if count > 0:
        print("-" * 60)
        print(f"Average MS-SSIM: {total_ms_ssim / count:.4f}")
        print(f"Average SSIM:    {total_ssim / count:.4f}")
        print("-" * 60)
    else:
        print("No images processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../Kodak", help="Path to dataset images")
    parser.add_argument("--weight_path", default="checkpoint_ucm_479/0379.ckpt", help="Path to checkpoint")
    parser.add_argument("--high", action="store_true", help="Use high bitrate model configuration")
    parser.add_argument("--post_processing", action="store_true", help="Use post processing")

    args = parser.parse_args()

    # 解决 OpenMP 冲突
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    calculate_metrics(args.data_path, args.weight_path, args.high, args.post_processing)