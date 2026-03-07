import os
import torch
import torch.nn as nn
import numpy as np
import math
import io  # 新增：用于内存操作
from PIL import Image
from torchvision import transforms

# === 尝试导入 compressai ===
try:
    from compressai.zoo import bmshj2018_hyperprior
except ImportError:
    print("请先安装库: pip install compressai")
    exit()

# ================= 配置区域 =================
# 你的验证集路径
DATA_PATH = r"D:\master\code\neural syntax\dataset\UCMerced_LandUse\val"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===========================================

def compute_psnr(a, b):
    # a, b 均为 [0, 1] 范围的 Tensor
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0: return 100
    return -10 * math.log10(mse)


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


# === 新增：JPEG 测试函数 ===
def eval_jpeg(img_pil, quality):
    # 1. 保存到内存 buffer，模拟压缩过程
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    file_size_bytes = buffer.tell()

    # 2. 计算 BPP
    w, h = img_pil.size
    bpp = (file_size_bytes * 8) / (w * h)

    # 3. 解码并转换格式用于计算 PSNR
    buffer.seek(0)
    img_rec = Image.open(buffer).convert('RGB')

    # 转为 Tensor [0, 1] 并移动到 device
    transform = transforms.ToTensor()
    img_orig_tensor = transform(img_pil).to(device)
    img_rec_tensor = transform(img_rec).to(device)

    psnr = compute_psnr(img_orig_tensor, img_rec_tensor)
    return bpp, psnr


def eval_model(model_func, quality, img_tensor):
    # 加载模型
    net = model_func(quality=quality, pretrained=True).to(device)
    net.eval()

    # 【修复部分】 处理尺寸问题
    _, _, h, w = img_tensor.size()
    p_h = (64 - (h % 64)) % 64
    p_w = (64 - (w % 64)) % 64

    if p_h > 0 or p_w > 0:
        img_padded = torch.nn.functional.pad(img_tensor, (0, p_w, 0, p_h), mode='reflect')
    else:
        img_padded = img_tensor

    with torch.no_grad():
        out = net(img_padded)

    x_hat = out['x_hat']
    x_hat = x_hat[:, :, :h, :w]  # 裁剪回原图尺寸

    bpp = compute_bpp(out)
    psnr = compute_psnr(img_tensor, x_hat)

    return bpp, psnr


def test_all_methods(data_path):
    print(f"正在对比 SOTA 模型 & JPEG (Device: {device})...")
    img_files = [f for f in os.listdir(data_path) if f.endswith(('.tif', '.png', '.jpg'))]

    transform = transforms.Compose([transforms.ToTensor()])

    # 存储结果
    results = {
        'JPEG (Q=30)': [],
        'Balle2018_Q3': [],
        'Balle2018_Q4': []
    }

    print(f"{'Image':<20} | {'Method':<15} | {'BPP':<8} | {'PSNR':<8}")
    print("-" * 65)

    count = 0
    for img_name in img_files:
        path = os.path.join(data_path, img_name)

        # 打开图片 (PIL 格式用于 JPEG)
        img_pil = Image.open(path).convert('RGB')
        # 转为 Tensor (用于 Ballé)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # 1. Test JPEG (Quality=30, 通常接近 0.8-1.0 bpp)
        bpp_j, psnr_j = eval_jpeg(img_pil, quality=30)
        results['JPEG (Q=30)'].append((bpp_j, psnr_j))

        # 2. Test Ballé 2018 (Quality 3)
        bpp3, psnr3 = eval_model(bmshj2018_hyperprior, 3, img_tensor)
        results['Balle2018_Q3'].append((bpp3, psnr3))

        # 3. Test Ballé 2018 (Quality 4)
        bpp4, psnr4 = eval_model(bmshj2018_hyperprior, 4, img_tensor)
        results['Balle2018_Q4'].append((bpp4, psnr4))

        count += 1
        if count <= 3:  # 只打印前3个详细信息
            print(f"{img_name[:20]:<20} | JPEG (Q=30)     | {bpp_j:<8.3f} | {psnr_j:<8.2f}")
            print(f"{'':<20} | Balle_Q3        | {bpp3:<8.3f} | {psnr3:<8.2f}")
            print(f"{'':<20} | Balle_Q4        | {bpp4:<8.3f} | {psnr4:<8.2f}")

    print("-" * 65)
    print("【FINAL RESULTS】 (Copy this to Gemini)")
    for key, vals in results.items():
        if len(vals) > 0:
            avg_bpp = np.mean([v[0] for v in vals])
            avg_psnr = np.mean([v[1] for v in vals])
            print(f"{key:<15} | Avg BPP: {avg_bpp:.4f} | Avg PSNR: {avg_psnr:.2f} dB")
        else:
            print(f"{key:<15} | No Data")


if __name__ == "__main__":
    test_all_methods(DATA_PATH)