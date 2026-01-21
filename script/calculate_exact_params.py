import torch
import torch.nn as nn


# ==========================================
# 1. 复制您 net.py 中的核心定义
# ==========================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)


class GDN(nn.Module):
    def __init__(self, ch):
        super(GDN, self).__init__()
        self.beta = nn.Parameter(torch.ones(ch))
        self.gamma = nn.Parameter(torch.eye(ch))  # 注意：gdn.py 里 gamma 是 ch*ch 的矩阵


# 模拟您的 analysisTransformModel (4层结构)
class RealEncoder(nn.Module):
    def __init__(self, N=192):  # 默认 N=192 (低码率模式)
        super(RealEncoder, self).__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(3, N, 5, 2, 0),
            GDN(N),
            DepthwiseSeparableConv(N, N, 5, 2, 0),
            GDN(N),
            DepthwiseSeparableConv(N, N, 5, 2, 0),
            GDN(N),
            DepthwiseSeparableConv(N, N, 5, 2, 0)
        )


# ==========================================
# 2. 精确计算
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # 情况 A: N=192 (标准设置)
    enc_192 = RealEncoder(N=192)
    params_192 = count_parameters(enc_192)

    # 情况 B: N=128 (如果为了更轻量)
    enc_128 = RealEncoder(N=128)
    params_128 = count_parameters(enc_128)

    print("-" * 40)
    print(f"【真实编码器结构 (4层 DSC + 3层 GDN)】")
    print("-" * 40)
    print(f"如果 N=192 (默认): 参数量 = {params_192 / 1e6:.4f} M")
    print(f"如果 N=128       : 参数量 = {params_128 / 1e6:.4f} M")
    print("-" * 40)

    # 对比 Ballé 2018 (单侧编码器约为 2.5 M)
    balle_params = 2.5  # M
    reduction = (1 - (params_192 / 1e6) / balle_params) * 100
    print(f"相比 Ballé (2.5M) 降低了: {reduction:.2f}%")


if __name__ == "__main__":
    main()