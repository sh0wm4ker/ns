import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 填入您的实验数据 (请替换为真实数据)
# ==========================================

# 您的算法 (Ours) - 至少跑 5 个点
bpp_ours = [0.15, 0.32, 0.55, 0.78, 1.05]
psnr_ours = [28.5, 30.8, 32.5, 33.9, 35.2]
msssim_ours = [0.92, 0.95, 0.97, 0.98, 0.99] # 如果用原值

# 对比算法 1: JPEG2000 (传统标准)
bpp_jp2k = [0.15, 0.32, 0.55, 0.78, 1.05]
psnr_jp2k = [26.2, 28.1, 29.8, 31.0, 32.1] # 通常比深度学习低 2-3dB
msssim_jp2k = [0.85, 0.89, 0.92, 0.94, 0.96]

# 对比算法 2: Ballé 2018 (深度学习基准) - 可选，如果没有数据可以用虚构的合理值占位
bpp_balle = [0.18, 0.35, 0.58, 0.80, 1.10]
psnr_balle = [28.8, 31.0, 32.8, 34.1, 35.5] # 通常比 Ours 略高或持平(因为它不轻量)
msssim_balle = [0.93, 0.96, 0.975, 0.985, 0.992]

# ==========================================
# 2. 绘图设置 (出版级风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体，显得学术
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # 一次画两张：PSNR 和 MS-SSIM

# --- 左图：Rate-Distortion (PSNR) ---
ax = axes[0]
# 画线：Ours 用最显眼的红色实线 + 五角星
ax.plot(bpp_ours, psnr_ours, color='#D62728', linestyle='-', linewidth=2.5,
        marker='*', markersize=12, label='Ours (ASGIC)')

# 画线：JPEG2000 用蓝色虚线 + 圆点
ax.plot(bpp_jp2k, psnr_jp2k, color='#1F77B4', linestyle='--', linewidth=2,
        marker='o', markersize=8, label='JPEG2000')

# 画线：Ballé 2018 用灰色点划线 + 三角
ax.plot(bpp_balle, psnr_balle, color='gray', linestyle='-.', linewidth=2,
        marker='^', markersize=8, label='Ballé 2018 (Baseline)')

ax.set_xlabel('Bitrate (bpp)', fontweight='bold')
ax.set_ylabel('PSNR (dB)', fontweight='bold')
ax.set_title('Rate-Distortion Performance (PSNR)', fontsize=14, y=1.02)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', frameon=True, shadow=True)

# --- 右图：Rate-Distortion (MS-SSIM) ---
ax = axes[1]
ax.plot(bpp_ours, msssim_ours, color='#D62728', linestyle='-', linewidth=2.5,
        marker='*', markersize=12, label='Ours (ASGIC)')
ax.plot(bpp_jp2k, msssim_jp2k, color='#1F77B4', linestyle='--', linewidth=2,
        marker='o', markersize=8, label='JPEG2000')
ax.plot(bpp_balle, msssim_balle, color='gray', linestyle='-.', linewidth=2,
        marker='^', markersize=8, label='Ballé 2018')

ax.set_xlabel('Bitrate (bpp)', fontweight='bold')
ax.set_ylabel('MS-SSIM', fontweight='bold')
ax.set_title('Perceptual Quality (MS-SSIM)', fontsize=14, y=1.02)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', frameon=True, shadow=True)

# 保存图片
plt.tight_layout()
plt.savefig('RD_Curves_Final.png', dpi=300, bbox_inches='tight') # 300dpi 高清保存
plt.show()