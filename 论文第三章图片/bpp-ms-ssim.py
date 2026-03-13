import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 中文显示设置
# =========================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 文件路径
# =========================
excel_file = "rd.xlsx"
output_file = "chapter3_bpp_msssim.png"

if not os.path.exists(excel_file):
    raise FileNotFoundError(f"未找到文件: {excel_file}")

# =========================
# 读取数据
# =========================
df = pd.read_excel(excel_file)

# 清理列名
df.columns = [str(c).strip() for c in df.columns]

# 仅保留第三章相关模型，忽略第四章模型
keep_models = ["Homo-DSC-5", "Rev-MG-DSC", "MG-DSC", "balle2018", "JPEG2000",]
df = df[df["model_name"].isin(keep_models)].copy()

# =========================
# 数值清洗函数
# =========================
def clean_numeric(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    match = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if match:
        return float(match[-1])
    return None

for col in ["bpp", "PSNR", "MS_SSIM"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# 删除缺失
df = df.dropna(subset=["bpp", "MS_SSIM"])

# 按 bpp 排序
df = df.sort_values(by=["model_name", "bpp"])

# =========================
# 颜色与线型
# =========================
style_map = {
    "Homo-DSC-5":   {"label": "Homo-DSC-5",  "marker": "o", "linestyle": "-",  "linewidth": 2.0},
    "Rev-MG-DSC":   {"label": "Rev-MG-DSC",  "marker": "s", "linestyle": "--", "linewidth": 2.0},
    "MG-DSC":       {"label": "MG-DSC",      "marker": "^", "linestyle": "-",  "linewidth": 2.4},
    "balle2018":    {"label": "Ballé2018",   "marker": "D", "linestyle": "-.", "linewidth": 2.0},
    "JPEG2000":     {"label": "JPEG2000",    "marker": "x", "linestyle": ":",  "linewidth": 2.0},
    "BPG":          {"label": "BPG",         "marker": "*", "linestyle": ":",  "linewidth": 2.0},
}

# =========================
# 绘图
# =========================
fig, ax = plt.subplots(figsize=(8.2, 5.6))

for model in keep_models:
    sub = df[df["model_name"] == model]
    if sub.empty:
        continue
    style = style_map.get(model, {})
    ax.plot(
        sub["bpp"],
        sub["MS_SSIM"],
        label=style.get("label", model),
        marker=style.get("marker", "o"),
        linestyle=style.get("linestyle", "-"),
        linewidth=style.get("linewidth", 2.0),
        markersize=6
    )

ax.set_xlabel("BPP", fontsize=12)
ax.set_ylabel("MS-SSIM", fontsize=12)
# ax.set_title("不同模型在测试集上的 bpp–MS-SSIM 曲线", fontsize=13)

ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"图像已保存为: {output_file}")