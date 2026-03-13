import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 中文显示设置
# =========================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 文件路径
# =========================
excel_file = "分场景收益统计.xlsx"
output_file = "不同典型场景下 MG-DSC 相对基线模型的 PSNR 收益统计.png"

if not os.path.exists(excel_file):
    raise FileNotFoundError(f"未找到文件: {excel_file}")

# =========================
# 读取Excel
# =========================
df = pd.read_excel(excel_file)

# 清理列名和第一列字符串
df.columns = [str(c).strip() for c in df.columns]
df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

# =========================
# 检查表格格式
# =========================
first_col = df.columns[0]
if first_col != "对比":
    raise ValueError(f"第一列应命名为“对比”，当前为: {first_col}")

scene_cols = df.columns[1:].tolist()
if len(scene_cols) == 0:
    raise ValueError("未检测到场景列，请检查表格格式。")

# =========================
# 提取数据
# =========================
def get_row_values(name_candidates):
    """
    根据候选名称列表匹配行，返回数值数组
    """
    for name in name_candidates:
        row = df[df["对比"] == name]
        if not row.empty:
            return row.iloc[0, 1:].astype(float).values
    raise ValueError(f"表中未找到以下任一行: {name_candidates}")

# 均值
mg_vs_homo = get_row_values(["MG-DSC vs Homo-DSC-5"])
mg_vs_rev  = get_row_values(["MG-DSC vs Rev-MG-DSC"])

# 误差棒
err_homo = get_row_values([
    "MG-DSC vs Homo-DSC-5的误差棒",
    "MG-DSC vs Homo-DSC-5 的误差棒"
])

err_rev = get_row_values([
    "MG-DSC vs Rev-MG-DC的误差棒",
    "MG-DSC vs Rev-MG-DC 的误差棒",
    "MG-DSC vs Rev-MG-DSC的误差棒",
    "MG-DSC vs Rev-MG-DSC 的误差棒"
])

# =========================
# 绘图
# =========================
x = np.arange(len(scene_cols))
width = 0.34

fig, ax = plt.subplots(figsize=(10, 5.5))

bars1 = ax.bar(
    x - width / 2,
    mg_vs_homo,
    width,
    yerr=err_homo,
    capsize=5,
    label="相对 Homo-DSC-5"
)

bars2 = ax.bar(
    x + width / 2,
    mg_vs_rev,
    width,
    yerr=err_rev,
    capsize=5,
    label="相对 Rev-MG-DSC"
)

# 0 基线
ax.axhline(0, color="black", linewidth=1)

# 坐标轴设置
ax.set_xticks(x)
ax.set_xticklabels(scene_cols, fontsize=11)
ax.set_ylabel("平均PSNR增益 / dB", fontsize=12)
ax.set_xlabel("场景类别", fontsize=12)

# 网格
ax.grid(axis="y", linestyle="--", alpha=0.35)

# 图例
ax.legend(fontsize=10, frameon=True)

# =========================
# 自动调整 y 轴范围
# 结合误差棒一起考虑
# =========================
all_lower = np.concatenate([mg_vs_homo - err_homo, mg_vs_rev - err_rev])
all_upper = np.concatenate([mg_vs_homo + err_homo, mg_vs_rev + err_rev])

ymin = np.min(all_lower) - 0.05
ymax = np.max(all_upper) + 0.06
ax.set_ylim(ymin, ymax)

# 用于控制标签与误差棒的相对间距
yrange = ymax - ymin
offset_pos = 0.012 * yrange   # 正值标签向上偏移
offset_neg = 0.014 * yrange   # 负值标签向下偏移

# =========================
# 数值标签
# 正值：放在误差棒顶端上方
# 负值：放在误差棒底端下方
# =========================
def add_labels(bars, values, errors):
    for b, v, e in zip(bars, values, errors):
        xpos = b.get_x() + b.get_width() / 2

        if v >= 0:
            y_text = v + e + offset_pos
            va = "bottom"
        else:
            y_text = v - e - offset_neg
            va = "top"

        ax.text(
            xpos,
            y_text,
            f"{v:.3f}",
            ha="center",
            va=va,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2)
        )

add_labels(bars1, mg_vs_homo, err_homo)
add_labels(bars2, mg_vs_rev, err_rev)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"图像已保存为: {output_file}")