import os

# ================= 关键修正 =================
# 必须在导入 torch 之前设置，解决 OpenMP 冲突报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ===========================================

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

# 引入你的模型定义
try:
    from model.net import Net
except ImportError:
    print("Error: 找不到 model.net，请确保脚本在项目根目录下运行")
    exit()

# ================= 配置区域 =================
# 1. 待测试的原图路径 (保持不变)
IMG_PATH = "val_pic/agricultural21.tif"

# 2. 你的模型权重路径 (已根据你的日志自动更新为 0259)
CKPT_PATH = "checkpoint_ucm_479/0379.ckpt"

# 3. JPEG2000 测试的压缩比率列表
JP2_RATIOS = [200, 150, 100, 80, 60, 40, 30, 20, 10]


# ===========================================

def calculate_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 10 * math.log10(max_pixel ** 2 / mse)
    return psnr


def get_neural_metrics(img_path, ckpt_path):
    """获取神经网络模型的 BPP 和 PSNR"""
    print(f"正在评估 Neural Syntax 模型: {ckpt_path} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 读取并预处理图片
    img_pil = Image.open(img_path).convert('RGB')
    w, h = img_pil.size

    # 补齐到 64 的倍数
    h_padded = h if h % 64 == 0 else (h // 64 + 1) * 64
    w_padded = w if w % 64 == 0 else (w // 64 + 1) * 64

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Padding
    pad_h = h_padded - h
    pad_w = w_padded - w
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # 归一化到 [-1, 1]
    input_tensor = img_tensor * 2.0 - 1.0

    # 2. 加载模型
    net = Net((1, h_padded, w_padded, 3), (1, h_padded, w_padded, 3), is_high=False, post_processing=False).to(device)

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        print(f"Warning: 找不到权重文件 {ckpt_path}，将使用随机初始化！")

    net.eval()

    # 3. 推理
    with torch.no_grad():
        bpp, mse, psnr = net(input_tensor, 'test')

    return bpp.item(), psnr.item()


def get_jpeg2000_metrics(img_path, ratios):
    """获取 JPEG2000 在不同压缩率下的 BPP 和 PSNR"""
    print("正在评估 JPEG2000 基准曲线...")
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    num_pixels = w * h

    results = []
    temp_file = "temp_jp2_test.jp2"
    gt_np = np.array(img).astype(np.float32)

    for r in ratios:
        try:
            # 保存为 JPEG2000
            img.save(temp_file, 'JPEG2000', quality_mode='rates', quality_layers=[r])

            # 计算真实 BPP
            file_size_bits = os.path.getsize(temp_file) * 8
            actual_bpp = file_size_bits / num_pixels

            # 计算 PSNR
            img_rec = Image.open(temp_file).convert('RGB')
            rec_np = np.array(img_rec).astype(np.float32)

            if rec_np.shape != gt_np.shape:
                min_h = min(rec_np.shape[0], gt_np.shape[0])
                min_w = min(rec_np.shape[1], gt_np.shape[1])
                rec_np = rec_np[:min_h, :min_w, :]
                gt_np_crop = gt_np[:min_h, :min_w, :]
                psnr = calculate_psnr(gt_np_crop, rec_np)
            else:
                psnr = calculate_psnr(gt_np, rec_np)

            results.append((actual_bpp, psnr))

        except Exception as e:
            print(f"  [跳过] JPEG2000 Ratio {r} 失败: {e}")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return sorted(results, key=lambda x: x[0])


def main():
    if not os.path.exists(IMG_PATH):
        print(f"Error: 找不到图片 {IMG_PATH}")
        return

    # 1. Neural Model
    n_bpp, n_psnr = get_neural_metrics(IMG_PATH, CKPT_PATH)
    print(f"\n>>> 你的模型结果: BPP = {n_bpp:.4f}, PSNR = {n_psnr:.2f} dB")

    # 2. JPEG2000
    jp2_results = get_jpeg2000_metrics(IMG_PATH, JP2_RATIOS)

    if not jp2_results:
        print("Error: JPEG2000 测试全部失败，请检查 Pillow 安装。")
        return

    # 3. 绘图
    plt.figure(figsize=(10, 7))

    jp2_bpps = [x[0] for x in jp2_results]
    jp2_psnrs = [x[1] for x in jp2_results]

    plt.plot(jp2_bpps, jp2_psnrs, 'b-o', linewidth=2, label='JPEG2000 (Baseline)')
    plt.plot(n_bpp, n_psnr, 'r*', markersize=18, label='Ours')

    plt.annotate(f'Ours\n({n_bpp:.2f}, {n_psnr:.2f})',
                 xy=(n_bpp, n_psnr), xytext=(n_bpp + 0.05, n_psnr - 1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title(f'Rate-Distortion Comparison', fontsize=14)
    plt.xlabel('Bit Per Pixel (BPP)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    save_path = 'rd_curve_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n对比图已保存为: {save_path}")
    # plt.show() # 如果不需要弹出窗口可注释掉


if __name__ == "__main__":
    main()