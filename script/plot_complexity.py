import matplotlib.pyplot as plt
import numpy as np

# ================= 数据录入 =================
# 格式: (Params_in_M, PSNR_dB, Label)

# 1. 你的模型 (0.24M 参数, 36.64 dB)
ours_data = (0.24, 36.64, "Ours")

# 2. Ballé 2018 (基准)
balle_data = (5.24, 36.19, "Ballé et al. [2018]")

# 3. 其他常见基准 (背景板)
minnen_data = (12.0, 37.2, "Minnen et al. [2018]")
cheng_data = (24.0, 37.8, "Cheng et al. [2020]")

# 整合数据
methods = [ours_data, balle_data, minnen_data, cheng_data]


# ===========================================

def main():
    # 设置画布大小 (4:3 比例)
    plt.figure(figsize=(8, 6))

    # 设置全局字体 (Times New Roman 风格)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # 提取数据
    params = [m[0] for m in methods]
    psnrs = [m[1] for m in methods]
    labels = [m[2] for m in methods]

    # 定义颜色和形状
    # Ours 使用深红色，其他使用深灰色或蓝色，保持沉稳
    colors = ['#d62728', '#1f77b4', '#7f7f7f', '#7f7f7f']
    markers = ['*', 's', 'o', 'D']
    sizes = [300, 120, 80, 80]  # Ours 略大以示区别，但不过分夸张

    # 绘制散点
    for i in range(len(methods)):
        # 绘制点
        plt.scatter(params[i], psnrs[i], c=colors[i], marker=markers[i], s=sizes[i],
                    edgecolors='k', linewidth=0.5, zorder=10)

        # 添加标签 (精细调整位置)
        # Ours: 在点下方
        if i == 0:
            ha = 'left'
            va = 'top'
            offset_x = 0
            offset_y = -0.2
            weight = 'bold'
        # Ballé: 在点下方
        elif i == 1:
            ha = 'right'
            va = 'top'
            offset_x = -0.5
            offset_y = -0.2
            weight = 'normal'
        # Others: 在点上方或左侧
        else:
            ha = 'right'
            va = 'bottom'
            offset_x = -1.0
            offset_y = 0.1
            weight = 'normal'

        plt.text(params[i] + offset_x, psnrs[i] + offset_y, labels[i],
                 fontsize=11, fontweight=weight, color=colors[i], ha=ha, va=va)

    # 坐标轴设置
    plt.xscale('log')  # 对数坐标是必须的

    # 网格线 (淡灰色)
    plt.grid(True, which="major", linestyle='--', alpha=0.4)
    plt.grid(True, which="minor", linestyle=':', alpha=0.2)

    # 轴标签
    plt.xlabel('Model Parameters (Millions, Log Scale)', fontweight='bold', fontsize=12)
    plt.ylabel('PSNR (dB)', fontweight='bold', fontsize=12)
    plt.title('Performance vs. Complexity Comparison', fontsize=14, y=1.02)

    # 设置范围 (留有余地)
    plt.xlim(0.1, 40)
    plt.ylim(35.5, 38.5)  # 根据数据微调，聚焦核心区域

    plt.tight_layout()
    save_path = "Figure4_Complexity_Clean.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 专业版复杂度对比图已生成: {save_path}")


if __name__ == "__main__":
    main()