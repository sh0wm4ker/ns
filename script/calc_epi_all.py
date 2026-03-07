import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import io

# ================= 配置区域 =================
DATA_PATH = r"../val_pic"
CKPT_PATH = r"../saves/4199.ckpt"
FIXED_BALLE_QUALITY = 3
# ===========================================

try:
    from compressai.zoo import bmshj2018_hyperprior
except ImportError:
    print("请 pip install compressai")
    exit()

try:
    from model.net import analysisTransformModel, synthesisTransformModel, Syntax_Model, \
        conv_generator, h_analysisTransformModel, h_synthesisTransformModel, \
        GaussianModel, PredictionModel_Context, PredictionModel_Syntax, \
        HAN, MeanShift
except ImportError:
    import sys

    sys.path.append("..")
    from model.net import analysisTransformModel, synthesisTransformModel, Syntax_Model, \
        conv_generator, h_analysisTransformModel, h_synthesisTransformModel, \
        GaussianModel, PredictionModel_Context, PredictionModel_Syntax, \
        HAN, MeanShift

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= 核心：EPI 计算函数 =================
def calc_epi(img1, img2):
    """
    计算边缘保持指数 (Edge Preservation Index, EPI)
    img1, img2: Tensor [B, 3, H, W] 或 [B, 1, H, W], 范围 0-255 或 0-1 均可
    """
    # 转灰度
    if img1.shape[1] == 3:
        g1 = 0.299 * img1[:, 0, :, :] + 0.587 * img1[:, 1, :, :] + 0.114 * img1[:, 2, :, :]
        g2 = 0.299 * img2[:, 0, :, :] + 0.587 * img2[:, 1, :, :] + 0.114 * img2[:, 2, :, :]
        g1, g2 = g1.unsqueeze(1), g2.unsqueeze(1)
    else:
        g1, g2 = img1, img2

    # 拉普拉斯算子
    kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]).to(img1.device)

    # 提取边缘
    edge1 = F.conv2d(g1, kernel, padding=1)
    edge2 = F.conv2d(g2, kernel, padding=1)

    # 计算相关系数
    e1_flat = edge1.view(edge1.shape[0], -1)
    e2_flat = edge2.view(edge2.shape[0], -1)

    numerator = (e1_flat * e2_flat).sum(dim=1)
    denominator = torch.sqrt((e1_flat ** 2).sum(dim=1) * (e2_flat ** 2).sum(dim=1))

    # 避免除零
    epi_val = numerator / (denominator + 1e-8)
    return epi_val.mean().item()


# ================= 模型定义 (NetFixed) =================
# 为了脚本独立运行，再次包含 NetFixed 类
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
    return transforms.ToTensor()(img_rec).unsqueeze(0).to(device)


def get_balle_tensor(img_pil, quality):
    net = bmshj2018_hyperprior(quality=quality, pretrained=True).to(device).eval()
    x = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
    h, w = x.shape[2], x.shape[3]
    p_h, p_w = (64 - h % 64) % 64, (64 - w % 64) % 64
    x_pad = F.pad(x, (0, p_w, 0, p_h), mode='reflect') if p_h + p_w > 0 else x
    with torch.no_grad():
        out = net(x_pad)
        x_hat = out['x_hat'][:, :, :h, :w]
    return x_hat.clamp(0, 1)


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
    return x_rec


def main():
    print(">>> 开始计算 EPI (Edge Preservation Index) ...")

    net = NetFixed((1, 256, 256, 3), (1, 256, 256, 3), is_high=False, post_processing=False).to(device)
    if os.path.exists(CKPT_PATH):
        try:
            net.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        except:
            pass
    net.eval()

    img_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(('.tif', '.png', '.jpg'))]

    epi_stats = {'JPEG': 0., 'Balle': 0., 'Ours': 0.}

    print("-" * 60)
    print(f"{'Image':<20} | {'Method':<8} | {'EPI':<8}")
    print("-" * 60)

    for img_name in img_files:
        path = os.path.join(DATA_PATH, img_name)
        img_pil = Image.open(path).convert('RGB')
        gt = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)  # [0, 1]

        # 1. JPEG
        rec_jpeg = get_jpeg_tensor(img_pil, quality=30)

        # 2. Ballé
        rec_balle = get_balle_tensor(img_pil, quality=FIXED_BALLE_QUALITY)

        # 3. Ours
        rec_ours = get_ours_tensor(img_pil, net)

        # 计算 EPI
        epi_j = calc_epi(gt, rec_jpeg)
        epi_b = calc_epi(gt, rec_balle)
        epi_o = calc_epi(gt, rec_ours)

        epi_stats['JPEG'] += epi_j
        epi_stats['Balle'] += epi_b
        epi_stats['Ours'] += epi_o

        print(f"{img_name[:20]:<20} | JPEG     | {epi_j:.4f}")
        print(f"{'':<20} | Balle    | {epi_b:.4f}")
        print(f"{'':<20} | Ours     | {epi_o:.4f}")
        print("-" * 60)

    n = len(img_files)
    print("\n" + "=" * 40)
    print("FINAL AVERAGE EPI RESULTS")
    print("=" * 40)
    print(f"JPEG (Q=30)  : {epi_stats['JPEG'] / n:.4f}")
    print(f"Ballé (Q={FIXED_BALLE_QUALITY})   : {epi_stats['Balle'] / n:.4f}")
    print(f"Ours         : {epi_stats['Ours'] / n:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()