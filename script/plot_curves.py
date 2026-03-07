import matplotlib.pyplot as plt
import re
import os
import numpy as np

# ================= 配置区域 =================
# 日志文件路径 (根据您之前的设置)
LOG_FILE = "/mnt/log/train_log_1768873924.txt"
# 图片保存路径
SAVE_PATH = "/mnt/log/training_result.png"


# ===========================================

def parse_and_plot():
    if not os.path.exists(LOG_FILE):
        print(f"❌ 错误：找不到日志文件: {LOG_FILE}")
        return

    epochs = []
    data = {
        'train_loss': [], 'val_loss': [],
        'train_bpp': [], 'val_bpp': [],
        'train_mse': [], 'val_mse': []
    }

    # 正则表达式匹配您的日志格式
    # 格式示例: [Epoch 4060] Train Loss: 3.4161 (Bpp: 0.7281 MSE: 268.7960) | Val Loss: 4.1228 (Bpp: 0.9053 MSE: 321.7424)
    pattern = re.compile(
        r'\[Epoch (\d+)\] Train Loss: ([\d\.]+) \(Bpp: ([\d\.]+) MSE: ([\d\.]+)\) \| Val Loss: ([\d\.]+) \(Bpp: ([\d\.]+) MSE: ([\d\.]+)\)'
    )

    print(f"📖 正在读取日志: {LOG_FILE} ...")

    with open(LOG_FILE, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                data['train_loss'].append(float(match.group(2)))
                data['train_bpp'].append(float(match.group(3)))
                data['train_mse'].append(float(match.group(4)))

                data['val_loss'].append(float(match.group(5)))
                data['val_bpp'].append(float(match.group(6)))
                data['val_mse'].append(float(match.group(7)))

    if not epochs:
        print("⚠️ 未提取到任何有效数据，请检查日志文件内容。")
        return

    print(f"✅ 成功提取 {len(epochs)} 条记录。正在绘图...")

    # 计算 PSNR (PSNR = 10 * log10(255^2 / MSE))
    # 防止 MSE 为 0 的情况
    train_mse_arr = np.array(data['train_mse'])
    val_mse_arr = np.array(data['val_mse'])

    train_psnr = 10 * np.log10((255 ** 2) / (train_mse_arr + 1e-10))
    val_psnr = 10 * np.log10((255 ** 2) / (val_mse_arr + 1e-10))

    # 创建画布
    plt.figure(figsize=(16, 10))

    # 1. Loss 曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, data['train_loss'], label='Train Loss', color='#1f77b4')
    plt.plot(epochs, data['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Bpp 曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, data['train_bpp'], label='Train Bpp', color='#2ca02c')
    plt.plot(epochs, data['val_bpp'], label='Val Bpp', color='#d62728', linestyle='--')
    plt.title('Bits Per Pixel (Bpp)')
    plt.xlabel('Epoch')
    plt.ylabel('Bpp')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. MSE 曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, data['train_mse'], label='Train MSE', color='#9467bd')
    plt.plot(epochs, data['val_mse'], label='Val MSE', color='#8c564b', linestyle='--')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. PSNR 曲线 (核心指标)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_psnr, label='Train PSNR', color='#e377c2')
    plt.plot(epochs, val_psnr, label='Val PSNR', color='#17becf', linestyle='--')
    plt.title('Peak Signal-to-Noise Ratio (PSNR)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"🎉 绘图完成！图片已保存至: {SAVE_PATH}")
    # 如果是在本地运行且支持GUI，可以取消下面这行的注释
    # plt.show()


if __name__ == "__main__":
    parse_and_plot()