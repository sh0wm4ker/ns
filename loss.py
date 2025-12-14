import matplotlib.pyplot as plt
import re
import argparse
import os


def plot_continuous_curves(log_path, output_dir):
    epochs = []
    losses = []
    bpps = []
    mses = []

    # 1. 读取并解析日志文件
    if not os.path.exists(log_path):
        print(f"Error: 找不到日志文件 {log_path}")
        return

    print(f"正在读取日志: {log_path} ...")

    with open(log_path, 'r') as f:
        for line in f:
            # 解析格式: [Epoch 0000 TRAIN] Loss: 141.1394 bpp: 1.0640 mse: 14007.5469
            match = re.search(r'Epoch (\d+) TRAIN] Loss: ([\d\.]+) bpp: ([\d\.]+) mse: ([\d\.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                bpps.append(float(match.group(3)))
                mses.append(float(match.group(4)))

    if not epochs:
        print("未在日志中找到有效数据。")
        return

    # 2. 【核心逻辑】智能拼接 Epoch
    # 无论你是否修改了 train.py 让 epoch 连续，这段代码都能处理
    # 如果检测到 Epoch 突然变小（说明重启了训练且计数重置），会自动加上偏移量

    corrected_epochs = []
    current_offset = 0
    last_raw_epoch = -1

    for i, raw_epoch in enumerate(epochs):
        # 如果当前读取的 epoch 比上一个还小（比如从 199 变成了 0），说明发生了断点续训
        if i > 0 and raw_epoch < last_raw_epoch:
            # 计算偏移量：让新的 0 接在旧的最后一个 epoch 后面
            # 新的起点应该是 corrected_epochs[-1] + 1
            # 所以 offset = (corrected_epochs[-1] + 1) - raw_epoch
            current_offset = corrected_epochs[-1] + 1 - raw_epoch
            print(f"[Info] 检测到训练重启 (Epoch {last_raw_epoch} -> {raw_epoch})，正在自动拼接曲线...")

        real_epoch = raw_epoch + current_offset
        corrected_epochs.append(real_epoch)
        last_raw_epoch = raw_epoch

    # 3. 创建保存目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. 绘图函数
    def save_plot(x, y, title, ylabel, filename, color):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=ylabel, color=color, linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Continuous Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"已保存: {save_path}")
        plt.close()

    # 5. 生成三张图
    save_plot(corrected_epochs, losses, 'Training Loss (Continuous)', 'Total Loss', 'loss_continuous.png', 'red')
    save_plot(corrected_epochs, bpps, 'BPP Curve (Continuous)', 'BPP (Bits Per Pixel)', 'bpp_continuous.png', 'blue')
    save_plot(corrected_epochs, mses, 'MSE Curve (Continuous)', 'MSE (Distortion)', 'mse_continuous.png', 'green')

    print("\n所有连续曲线图已生成完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径指向 checkpoint_ucm，你可以根据实际情况修改
    parser.add_argument("--log_path", type=str, default="checkpoint_ucm/train_log.txt", help="Path to log file")
    parser.add_argument("--output_dir", type=str, default="output_plots_continuous", help="Output directory")

    args = parser.parse_args()

    plot_continuous_curves(args.log_path, args.output_dir)