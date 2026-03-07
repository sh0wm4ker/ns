import os

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import math

# ================= 配置区域 =================
# 1. 验证集路径 (请修改为您实际的验证集文件夹)
DATA_PATH = r"../val_pic"  # 或者绝对路径

# 2. 您的模型权重
CKPT_PATH = r"../saves/4199.ckpt"

# 3. 对比设置
#    Ballé 的质量等级 (1-8)，建议 3 或 4
FIXED_BALLE_QUALITY = 3
# ===========================================

try:
    from compressai.zoo import bmshj2018_hyperprior
except ImportError:
    print("❌ 错误: 请先安装 compressai (pip install compressai)")
    exit()

# 尝试导入您的模型定义，如果失败则使用内置的 NetFixed
try:
    from model.net import analysisTransformModel, synthesisTransformModel, Syntax_Model, \
        conv_generator, h_analysisTransformModel, h_synthesisTransformModel, \
        GaussianModel, PredictionModel_Context, PredictionModel_Syntax, \
        HAN, MeanShift
except ImportError:
    # 假设脚本在 script 目录下运行，尝试添加父目录到 path
    import sys

    sys.path.append("..")
    from model.net import analysisTransformModel, synthesisTransformModel, Syntax_Model, \
        conv_generator, h_analysisTransformModel, h_synthesisTransformModel, \
        GaussianModel, PredictionModel_Context, PredictionModel_Syntax, \
        HAN, MeanShift

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= SSIM / MS-SSIM 计算函数 =================
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
    if not X.shape == Y.shape: return 0.0
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
    return ms_ssim_val.item()


def calc_single_ssim(X, Y):
    win = _fspecial_gauss_1d(11, 1.5).repeat(3, 1, 1, 1)
    val, _ = _ssim(X, Y, win=win, data_range=255.0)
    return val.item()


# ================= 模型定义 (复制自 generate_figure.py 以确保兼容) =================
class BlockSampleFixed(nn.Module):
    def __init__(self, in_shape, masked=True):
        super(BlockSampleFixed, self).__init__()
        self.masked = masked
        dim = in_shape[1]
        flt = np.zeros((dim * 16, dim, 7, 7), dtype=np.float32)
        for i in range(0, 4):
            for j in range(0, 4):
                if self.masked and i == 3 and (j == 2 or j == 3): break
                for k in range(0, dim):
                    s = k * 16 + i * 4 + j
                    flt[s, k, i, j + 1] = 1
        self.register_buffer('sample_filter', torch.from_numpy(flt).float())

    def forward(self, inputs):
        t = F.conv2d(inputs, self.sample_filter, padding=3)
        b, c, h, w = inputs.size()
        return t.contiguous().view(b, c, 4, 4, h, w).permute(0, 4, 5, 1, 2, 3).contiguous().view(b * h * w, c, 4, 4)


class NetFixed(nn.Module):
    def __init__(self, train_size, test_size, is_high, post_processing):
        super(NetFixed, self).__init__()
        self.is_high = is_high
        N, M = (384, 32) if is_high else (192, 16)
        self.M, self.N = M, N
        self.a_model = analysisTransformModel(3, [N, N, N, N])
        self.s_model = synthesisTransformModel(N - M, [N, N, N, M])
        self.syntax_model = Syntax_Model(M, M)
        self.conv_weights_gen = conv_generator(in_dim=M, out_dim=M)
        self.ha_model = h_analysisTransformModel(N, [N, N, N], [1, 2, 2])
        self.hs_model = h_synthesisTransformModel(N, [N, N, N], [2, 2, 1])
        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z3 = GaussianModel()
        self.entropy_bottleneck_z3_syntax = GaussianModel()
        self.v_z2_sigma = nn.Parameter(torch.ones((1, N, 1, 1), dtype=torch.float32))
        self.register_parameter('z2_sigma', self.v_z2_sigma)
        self.prediction_model = PredictionModel_Context(in_dim=2 * N - M, dim=N, outdim=(N - M) * 2)
        self.prediction_model_syntax = PredictionModel_Syntax(in_dim=N, dim=M, outdim=M * 2)
        b, h, w, c = train_size
        tb, th, tw, tc = test_size
        self.test_y_sampler = BlockSampleFixed((b, N - M, th // 8, tw // 8))
        self.test_h_sampler = BlockSampleFixed((b, N, th // 8, tw // 8), False)
        self.HAN = HAN(is_high=self.is_high)
        self.conv_weights_gen_HAN = conv_generator(in_dim=M, out_dim=64)
        self.add_mean = MeanShift(1.0, (0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0), 1)

    def batch_conv(self, weights, inputs):
        b, ch, _, _ = inputs.shape
        _, ch_out, _, k, _ = weights.shape
        weights = weights.reshape(b * ch_out, ch, k, k)
        inputs = torch.cat(torch.split(inputs, 1, dim=0), dim=1)
        out = F.conv2d(inputs, weights, stride=1, padding=0, groups=b)
        return torch.cat(torch.split(out, ch_out, dim=1), dim=0)


# ================= 推理工具 =================
def get_jpeg_tensor(img_pil, quality):
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img_rec = Image.open(buffer).convert('RGB')
    return transforms.ToTensor()(img_rec).unsqueeze(0).to(device) * 255.0


def get_balle_tensor(img_pil, quality):
    net = bmshj2018_hyperprior(quality=quality, pretrained=True).to(device).eval()
    x = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
    h, w = x.shape[2], x.shape[3]
    p_h, p_w = (64 - h % 64) % 64, (64 - w % 64) % 64
    x_pad = F.pad(x, (0, p_w, 0, p_h), mode='reflect') if p_h + p_w > 0 else x
    with torch.no_grad():
        out = net(x_pad)
        x_hat = out['x_hat'][:, :, :h, :w]
    return x_hat.clamp(0, 1) * 255.0


def get_ours_tensor(img_pil, net):
    x = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
    h, w = x.shape[2], x.shape[3]
    h_pad, w_pad = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
    x_in = F.pad(x * 2 - 1, (0, w_pad - w, 0, h_pad - h), mode='constant', value=0)
    with torch.no_grad():
        z3 = net.a_model(x_in)
        z3_syn = torch.round(net.syntax_model(z3[:, :net.M]))
        z3_con = torch.round(z3[:, net.M:])
        x_rec = net.batch_conv(net.conv_weights_gen(z3_syn), net.s_model(z3_con))
        x_rec = torch.clamp((x_rec + 1) / 2, 0, 1)[:, :, :h, :w]
    return x_rec * 255.0


def main():
    print(">>> 开始计算 SSIM / MS-SSIM 对比数据...")

    # 加载 Ours 模型
    net = NetFixed((1, 256, 256, 3), (1, 256, 256, 3), is_high=False, post_processing=False).to(device)
    if os.path.exists(CKPT_PATH):
        try:
            net.load_state_dict(torch.load(CKPT_PATH, map_location=device))
            print(f"Loaded Ours: {CKPT_PATH}")
        except:
            print("Warning: Checkpoint load failed, utilizing random weights.")
    net.eval()

    img_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(('.tif', '.png', '.jpg'))]
    if not img_files:
        print(f"Error: No images found in {DATA_PATH}")
        return

    # 统计器
    stats = {
        'JPEG': {'ssim': 0., 'ms_ssim': 0.},
        'Balle': {'ssim': 0., 'ms_ssim': 0.},
        'Ours': {'ssim': 0., 'ms_ssim': 0.}
    }

    print("-" * 80)
    print(f"{'Image':<20} | {'Method':<8} | {'SSIM':<8} | {'MS-SSIM':<8}")
    print("-" * 80)

    for img_name in img_files:
        path = os.path.join(DATA_PATH, img_name)
        img_pil = Image.open(path).convert('RGB')

        # GT Tensor (0-255)
        gt = transforms.ToTensor()(img_pil).unsqueeze(0).to(device) * 255.0

        # 1. JPEG (Q=30, approx 0.8 bpp)
        # 为了公平，您可以调整 quality 使得 BPP 接近 Ours
        rec_jpeg = get_jpeg_tensor(img_pil, quality=30)

        # 2. Ballé
        rec_balle = get_balle_tensor(img_pil, quality=FIXED_BALLE_QUALITY)

        # 3. Ours
        rec_ours = get_ours_tensor(img_pil, net)

        # 计算指标
        # JPEG
        s_j = calc_single_ssim(gt, rec_jpeg)
        ms_j = ms_ssim(gt, rec_jpeg)
        stats['JPEG']['ssim'] += s_j
        stats['JPEG']['ms_ssim'] += ms_j

        # Ballé
        s_b = calc_single_ssim(gt, rec_balle)
        ms_b = ms_ssim(gt, rec_balle)
        stats['Balle']['ssim'] += s_b
        stats['Balle']['ms_ssim'] += ms_b

        # Ours
        s_o = calc_single_ssim(gt, rec_ours)
        ms_o = ms_ssim(gt, rec_ours)
        stats['Ours']['ssim'] += s_o
        stats['Ours']['ms_ssim'] += ms_o

        print(f"{img_name[:20]:<20} | JPEG     | {s_j:.4f}   | {ms_j:.4f}")
        print(f"{'':<20} | Balle    | {s_b:.4f}   | {ms_b:.4f}")
        print(f"{'':<20} | Ours     | {s_o:.4f}   | {ms_o:.4f}")
        print("-" * 80)

    n = len(img_files)
    print("\n" + "=" * 40)
    print("FINAL AVERAGE RESULTS")
    print("=" * 40)
    print(f"JPEG (Q=30)  : SSIM={stats['JPEG']['ssim'] / n:.4f}, MS-SSIM={stats['JPEG']['ms_ssim'] / n:.4f}")
    print(
        f"Ballé (Q={FIXED_BALLE_QUALITY})   : SSIM={stats['Balle']['ssim'] / n:.4f}, MS-SSIM={stats['Balle']['ms_ssim'] / n:.4f}")
    print(f"Ours         : SSIM={stats['Ours']['ssim'] / n:.4f}, MS-SSIM={stats['Ours']['ms_ssim'] / n:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()