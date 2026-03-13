import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 中文显示设置
# =========================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# =========================
# 文件路径
# =========================
complexity_file = "模型复杂度.xlsx"
rd_file = "rd.xlsx"
output_file = "复杂度-性能折中图.png"

if not os.path.exists(complexity_file):
    raise FileNotFoundError(f"未找到文件: {complexity_file}")
if not os.path.exists(rd_file):
    raise FileNotFoundError(f"未找到文件: {rd_file}")

# =========================
# 名称标准化
# =========================
def normalize_model_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    if "homo" in s and "dsc" in s:
        return "Homo-DSC-5"
    if "rev" in s and "mg" in s and "dsc" in s:
        return "Rev-MG-DSC"
    if s == "mgdsc" or ("mg" in s and "dsc" in s and "rev" not in s and "homo" not in s):
        return "MG-DSC"
    if "balle" in s or "ballé" in s:
        return "Ballé2018"
    return str(name).strip()

# =========================
# 数值清洗
# 兼容类似 ".    32.51" 的字符串
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

# =========================
# 读取复杂度表
# =========================
df_c = pd.read_excel(complexity_file)
df_c.columns = [str(c).strip() for c in df_c.columns]
df_c["model_name"] = df_c["model_name"].apply(normalize_model_name)

keep_models = ["Homo-DSC-5", "Rev-MG-DSC", "MG-DSC", "Ballé2018"]
df_c = df_c[df_c["model_name"].isin(keep_models)].copy()

for col in df_c.columns:
    if col not in ["model_name", "model_type"]:
        df_c[col] = df_c[col].apply(clean_numeric)

# =========================
# 读取 RD 表
# =========================
df_r = pd.read_excel(rd_file)
df_r.columns = [str(c).strip() for c in df_r.columns]
df_r["model_name"] = df_r["model_name"].apply(normalize_model_name)
df_r = df_r[df_r["model_name"].isin(keep_models)].copy()

for col in ["bpp", "PSNR", "MS_SSIM"]:
    if col in df_r.columns:
        df_r[col] = df_r[col].apply(clean_numeric)

df_r = df_r.dropna(subset=["PSNR"])

# 计算平均 PSNR
psnr_avg = (
    df_r.groupby("model_name", as_index=False)["PSNR"]
    .mean()
    .rename(columns={"PSNR": "avg_PSNR"})
)

# =========================
# 合并
# =========================
df = pd.merge(df_c, psnr_avg, on="model_name", how="inner")

model_order = ["Homo-DSC-5", "Rev-MG-DSC", "MG-DSC", "Ballé2018"]
df["order"] = df["model_name"].apply(lambda x: model_order.index(x))
df = df.sort_values("order").reset_index(drop=True)

# =========================
# 绘图样式
# =========================
marker_map = {
    "Homo-DSC-5": "o",
    "Rev-MG-DSC": "s",
    "MG-DSC": "^",
    "Ballé2018": "D",
}

# 为标签设置偏移量，避免重叠
offsets_left = {
    "Homo-DSC-5": (0.01, 0.03),
    "Rev-MG-DSC": (0.01, 0.03),
    "MG-DSC": (0.01, 0.03),
    "Ballé2018": (-0.3, -0.05),
}

offsets_right = {
    "Homo-DSC-5": (0.15, 0.03),
    "Rev-MG-DSC": (0.15, 0.03),
    "MG-DSC": (0.15, 0.03),
    "Ballé2018": (-0.3, -0.05),
}

# =========================
# 绘图
# =========================
fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1))

# 统一边框与网格
for ax in axes:
    ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

# ---------- (a) 编码端 ----------
ax = axes[0]
for _, row in df.iterrows():
    name = row["model_name"]
    x = row["encoder_MACs (G)"]
    y = row["avg_PSNR"]

    ax.scatter(
        x, y,
        s=140,
        marker=marker_map[name],
        edgecolors="white",
        linewidths=1.1,
        zorder=3
    )

    dx, dy = offsets_left[name]
    ax.text(
        x + dx, y + dy,
        name,
        fontsize=10,
        ha="left",
        va="bottom"
    )

ax.set_xlabel("编码端 MACs / G", fontsize=12)
ax.set_ylabel("平均 PSNR / dB", fontsize=12)
ax.text(0.5, -0.16, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=11)

# ---------- (b) 整体 ----------
ax = axes[1]
for _, row in df.iterrows():
    name = row["model_name"]
    x = row["total_MACs (G)"]
    y = row["avg_PSNR"]

    ax.scatter(
        x, y,
        s=140,
        marker=marker_map[name],
        edgecolors="white",
        linewidths=1.1,
        zorder=3
    )

    dx, dy = offsets_right[name]
    ax.text(
        x + dx, y + dy,
        name,
        fontsize=10,
        ha="left",
        va="bottom"
    )

ax.set_xlabel("整体 MACs / G", fontsize=12)
ax.set_ylabel("平均 PSNR / dB", fontsize=12)
ax.text(0.5, -0.16, "(b)", transform=ax.transAxes, ha="center", va="top", fontsize=11)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("用于绘图的数据如下：")
print(df[[
    "model_name",
    "encoder_MACs (G)",
    "total_MACs (G)",
    "avg_PSNR"
]])
print(f"\n图像已保存为: {output_file}")