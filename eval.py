import argparse
import os
import time
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import numpy as np
from torch import optim

# 引入你的模型定义
from model.net import Net


# ==========================================
# [新增] SSIM & MS-SSIM 计算工具函数
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

    mcs = torch.stack(mcs)
    mcs[-1] = ssim_val
    ms_ssim_val = torch.prod(mcs ** weights)
    return ms_ssim_val


def calc_psnr(mse):
    return 10 * torch.log10(255 * 255 / mse)


# ==========================================
# 验证主逻辑
# ==========================================
def val(data_path, weight_path, lmbda, is_high, post_processing, pre_processing, tune_iter):
    # 用于统计平均指标
    list_eval_bpp = 0.
    list_v_psnr = 0.
    list_ssim = 0.
    list_ms_ssim = 0.
    cnt = 0
    sum_time = 0.

    # 准备 SSIM 计算用的 window
    win_ssim = _fspecial_gauss_1d(11, 1.5).repeat(3, 1, 1, 1)

    print(f"{'Image':<30} | {'BPP':<8} | {'PSNR':<8} | {'SSIM':<8} | {'MS-SSIM':<8} | {'Time':<8}")
    print("-" * 85)

    image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

    for img_name in image_files:
        # 1. 数据加载与预处理
        img_path = os.path.join(data_path, img_name)
        data = Image.open(img_path).convert('RGB')
        data = tv.transforms.ToTensor()(data)

        _, h, w = data.shape

        # Padding 到 64 的倍数 (适配模型结构)
        h_padded = h if h % 64 == 0 else (h // 64 + 1) * 64
        w_padded = w if w % 64 == 0 else (w // 64 + 1) * 64
        padding_h = h_padded - h
        padding_w = w_padded - w

        data = F.pad(data, (0, padding_w, 0, padding_h), mode='constant', value=0)  # [C, H_pad, W_pad]
        data = data.unsqueeze(0)  # [1, C, H, W]

        # 归一化到 [-1, 1]
        data = data * 2.0 - 1.0

        # 2. 初始化模型
        # 注意：由于原代码每次循环都重新 init Net，这里保持一致
        net = Net((1, h_padded, w_padded, 3), (1, h_padded, w_padded, 3), is_high, post_processing).cuda()
        net.load_state_dict(torch.load(weight_path), strict=True)

        begin_time = time.time()

        # 3. 在线微调 (Pre-processing) 逻辑
        if pre_processing:
            opt_enc = optim.Adam(net.a_model.parameters(), lr=1e-5)
            sch = optim.lr_scheduler.MultiStepLR(opt_enc, [50], 0.5)

            net.post_processing = False
            for iters in range(tune_iter):
                train_bpp, train_mse = net(data.cuda(), 'train')
                train_loss = lmbda * train_mse + train_bpp
                train_loss = train_loss.mean()
                opt_enc.zero_grad()
                train_loss.backward()
                opt_enc.step()
                sch.step()
            net.post_processing = post_processing

        # 4. 推理 (Inference)
        with torch.no_grad():
            data_cuda = data.cuda()

            # A. 获取 BPP (调用 net 原有逻辑)
            eval_bpp, _, _ = net(data_cuda, 'test')

            # B. 手动执行前向传播获取重建图像 (为了计算 SSIM)
            # -------------------------------------------------
            # Encoder
            z3 = net.a_model(data_cuda)
            z3_syntax = z3[:, :net.M, :, :]
            z3_syntax = net.syntax_model(z3_syntax)
            z3_content = z3[:, net.M:, :, :]

            # Quantize
            z3_content_hat = torch.round(z3_content)
            z3_syntax_hat = torch.round(z3_syntax)

            # Decoder (Content)
            x_tilde = net.s_model(z3_content_hat)

            # Decoder (Weights & Modulation)
            conv_weights = net.conv_weights_gen(z3_syntax_hat)
            x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

            # Post Processing
            if post_processing:
                x_tilde = net.HAN(x_tilde_bf)
                conv_weights_han = net.conv_weights_gen_HAN(z3_syntax_hat)
                x_tilde = net.batch_conv(conv_weights_han, x_tilde)
                x_tilde = net.add_mean(x_tilde)
            else:
                x_tilde = x_tilde_bf

            # -------------------------------------------------

            # 还原到 0-255
            # GT (原始图像)
            gt = torch.clamp((data_cuda + 1) * 127.5, 0, 255).round()
            # Recon (重建图像)
            x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255).round()

            # 去除 Padding (只计算有效区域)
            gt = gt[:, :, :h, :w]
            x_hat = x_hat[:, :, :h, :w]

            # 计算指标
            # 1. PSNR (使用最新的 x_hat 计算，确保准确)
            mse_val = F.mse_loss(x_hat, gt)
            v_psnr = calc_psnr(mse_val)

            # 2. MS-SSIM
            v_ms_ssim = ms_ssim(gt, x_hat, data_range=255.0)

            # 3. SSIM
            v_ssim, _ = _ssim(gt, x_hat, win=win_ssim, data_range=255.0)

        end_time = time.time()
        duration = end_time - begin_time
        sum_time += duration

        # 记录数据
        current_bpp = eval_bpp.mean().item()
        current_psnr = v_psnr.item()
        current_ssim = v_ssim.item()
        current_ms_ssim = v_ms_ssim.item()

        list_eval_bpp += current_bpp
        list_v_psnr += current_psnr
        list_ssim += current_ssim
        list_ms_ssim += current_ms_ssim
        cnt += 1

        print(
            f"{img_name[:30]:<30} | {current_bpp:<8.4f} | {current_psnr:<8.4f} | {current_ssim:<8.4f} | {current_ms_ssim:<8.4f} | {duration:<8.4f}")

    if cnt > 0:
        print("-" * 85)
        print(
            f"{'AVERAGE':<30} | {list_eval_bpp / cnt:<8.4f} | {list_v_psnr / cnt:<8.4f} | {list_ssim / cnt:<8.4f} | {list_ms_ssim / cnt:<8.4f} | {sum_time / cnt:<8.4f}")
        print("-" * 85)
    else:
        print("No images found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path", default="../Kodak", help="Directory of Testset Images")
    parser.add_argument("--weight_path", default="./test.ckpt", help="Path of Checkpoint")
    parser.add_argument("--high", action="store_true", help="Using High Bitrate Model")
    parser.add_argument("--post_processing", action="store_true", help="Using Post Processing")
    parser.add_argument("--pre_processing", action="store_true", help="Using Pre Processing (Online Finetuning)")
    parser.add_argument("--lambda", type=float, default=0.01, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument("--tune_iter", type=int, default=100, help="Finetune Iteration")

    args = parser.parse_args()

    # 解决潜在的 OpenMP 重复初始化错误
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    val(args.data_path, args.weight_path, args.lmbda, args.high, args.post_processing, args.pre_processing,
        args.tune_iter)