import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# 从您的项目模型文件中引入 Net
try:
    from model.net import Net
except ImportError:
    print("Error: 找不到 model.net，请确保脚本在项目根目录下运行")
    exit()


# ==========================================
# 辅助函数：指标计算
# ==========================================
def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _ssim(X, Y, win, data_range=255.0):
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = F.conv2d(X, win, stride=1, padding=0, groups=X.shape[1])
    mu2 = F.conv2d(Y, win, stride=1, padding=0, groups=Y.shape[1])

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(X * X, win, groups=X.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(Y * Y, win, groups=Y.shape[1]) - mu2_sq
    sigma12 = F.conv2d(X * Y, win, groups=X.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def calc_psnr(mse):
    if mse == 0: return 100.0
    return (10 * torch.log10(255.0 * 255.0 / mse)).item()


# ==========================================
# 推理函数
# ==========================================
def run_inference(ckpt_path, img_tensor_padded, original_h, original_w, is_high, post_processing):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_pad, w_pad = img_tensor_padded.shape[2], img_tensor_padded.shape[3]

    net = Net((1, h_pad, w_pad, 3), (1, h_pad, w_pad, 3), is_high, post_processing).to(device)

    if os.path.exists(ckpt_path):
        # 增加了 weights_only=True 以消除 PyTorch 警告
        net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"成功加载权重: {ckpt_path}")
    else:
        print(f"错误: 找不到权重文件 {ckpt_path}")
        return None, 0, 0

    net.eval()
    with torch.no_grad():
        input_tensor = (img_tensor_padded * 2.0 - 1.0).to(device)

        z3 = net.a_model(input_tensor)
        z3_syntax = net.syntax_model(z3[:, :net.M, :, :])
        z3_content = z3[:, net.M:, :, :]

        z3_syntax_hat = torch.round(z3_syntax)
        z3_content_hat = torch.round(z3_content)

        conv_weights = net.conv_weights_gen(z3_syntax_hat)
        x_tilde = net.s_model(z3_content_hat)
        x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

        if post_processing:
            x_tilde_bf = net.HAN(x_tilde_bf)
            conv_weights_han = net.conv_weights_gen_HAN(z3_syntax_hat)
            x_tilde_bf = net.batch_conv(conv_weights_han, x_tilde_bf)
            x_tilde_bf = net.add_mean(x_tilde_bf)

        recon = torch.clamp((x_tilde_bf + 1) * 127.5, 0, 255).round()
        gt = torch.clamp((input_tensor + 1) * 127.5, 0, 255).round()

        recon = recon[:, :, :original_h, :original_w]
        gt = gt[:, :, :original_h, :original_w]

        mse_val = F.mse_loss(recon, gt)
        psnr = calc_psnr(mse_val)

        win = _fspecial_gauss_1d(11, 1.5).repeat(3, 1, 1, 1).to(device)
        ssim = _ssim(gt, recon, win)

    return recon.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8), psnr, ssim


# ==========================================
# 辅助函数：绘制单列（原图 + 放大图）
# ==========================================
def plot_column(ax_full, ax_crop, img_array, title, crop_box):
    x, y, w, h = crop_box

    # 1. 显示大图
    ax_full.imshow(img_array)
    ax_full.set_title(title, fontsize=12)
    ax_full.axis('off')

    # 2. 在大图上画红色矩形框
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax_full.add_patch(rect)

    # 3. 截取局部图像并显示
    crop_img = img_array[y:y + h, x:x + w]
    ax_crop.imshow(crop_img)
    # 为了突出显示，可以给放大图也加上红边框
    for spine in ax_crop.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)
    ax_crop.set_xticks([])
    ax_crop.set_yticks([])


# ==========================================
# 主运行逻辑
# ==========================================
def compare_two_ckpts(img_path, ckpt1, ckpt2, is_high, post_processing, crop_box):
    # 1. 图像预处理
    img_pil = Image.open(img_path).convert('RGB')
    w, h = img_pil.size
    img_np = np.array(img_pil)

    # 补齐到 64 的倍数
    h_padded = h if h % 64 == 0 else (h // 64 + 1) * 64
    w_padded = w if w % 64 == 0 else (w // 64 + 1) * 64

    transform = T.ToTensor()
    img_tensor = transform(img_pil).unsqueeze(0)
    img_tensor_padded = F.pad(img_tensor, (0, w_padded - w, 0, h_padded - h), mode='constant', value=0)

    # 2. 运行模型 1 和 2
    res1, psnr1, ssim1 = run_inference(ckpt1, img_tensor_padded, h, w, is_high, post_processing)
    res2, psnr2, ssim2 = run_inference(ckpt2, img_tensor_padded, h, w, is_high, post_processing)

    # 3. 可视化对比 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 原图列
    plot_column(axes[0, 0], axes[1, 0], img_np, "Original Image", crop_box)

    # 模型1列
    if res1 is not None:
        title1 = f"Model: {os.path.basename(ckpt1)}\nPSNR: {psnr1:.2f}dB | SSIM: {ssim1:.4f}"
        plot_column(axes[0, 1], axes[1, 1], res1, title1, crop_box)

    # 模型2列
    if res2 is not None:
        title2 = f"Model: {os.path.basename(ckpt2)}\nPSNR: {psnr2:.2f}dB | SSIM: {ssim2:.4f}"
        plot_column(axes[0, 2], axes[1, 2], res2, title2, crop_box)

    plt.tight_layout()
    save_path = "model_comparison_with_zoom.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图（带局部放大）已保存至: {save_path}")


if __name__ == "__main__":
    # ---------------- 配置参数 ----------------
    TEST_IMAGE = "../val_pic/airplane00.tif"
    CKPT_1 = "../saves/4199.ckpt"
    CKPT_2 = "../saves/0379.ckpt"

    IS_HIGH = False
    POST_PROCESS = False

    # 【新增】定义局部放大的区域坐标
    # 格式为: (x坐标, y坐标, 宽度, 高度)
    # 请根据您的测试图片内容自行调整这些数字，选中最容易看出画质差异的地方（如纹理密集区、文字边缘等）
    CROP_BOX = (150, 100, 50, 50)
    # ------------------------------------------

    compare_two_ckpts(TEST_IMAGE, CKPT_1, CKPT_2, IS_HIGH, POST_PROCESS, CROP_BOX)