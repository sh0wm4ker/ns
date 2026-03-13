import os
import tempfile
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio

# 解决可能存在的 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========= 你的模型 =========
try:
    from model.net import Net
except ImportError:
    print("Error: 找不到 model.net，请确保脚本在项目根目录下运行")
    raise

# =========================================================
# 1. 用户配置
# =========================================================

# ---------- 输入图像 ----------
IMG_PATH = r"../val_pic/airplane03.tif"

# ---------- 输出路径 ----------
OUT_FIG_PATH = r"毕业论文主观重建图.png"

# ---------- 裁剪框 ----------
CROP_CENTER = (128, 128)
CROP_WIDTH = 80
CROP_HEIGHT = 80

# ---------- OpenJPEG ----------
OPJ_COMPRESS = r"D:\tool\openjpeg-v2.4.0-windows-x64\openjpeg-v2.4.0-windows-x64\bin\opj_compress.exe"
OPJ_DECOMPRESS = r"D:\tool\openjpeg-v2.4.0-windows-x64\openjpeg-v2.4.0-windows-x64\bin\opj_decompress.exe"

# 你通过修改这个列表控制 JPEG2000 压缩率
JP2_RATIOS = [64]

# ---------- 模型权重 ----------
MODEL_SPECS = {
    "Ballé2018": {
        "ckpt": r"C:\Users\魏新烨\Desktop\sij\原始模型\0.067\balle0067 1599.ckpt",
        "is_high": False,
        "post_processing": False,
    },
    "MG-DSC": {
        "ckpt": r"C:\Users\魏新烨\Desktop\sij\原始模型\0.067\mgdsc0067 1499.ckpt",
        "is_high": False,
        "post_processing": False,
    },
}

# =========================================================
# bpp / PSNR 显示模式开关
# False: 默认自动计算
# True : 使用手动填写数值
# =========================================================
USE_MANUAL_METRICS = True

MANUAL_METRICS = {
    "JPEG2000": {
        "bpp": 0.543,
        "PSNR": 31.66,
    },
    "Ballé2018": {
        "bpp": 0.562,
        "PSNR": 33.61,
    },
    "MG-DSC": {
        "bpp": 0.559,
        "PSNR": 33.45,
    }
}

# ---------- 显示控制 ----------
DRAW_ROI_ON_GT = True
SHOW_GT_CROP = True   # 是否显示单独的 GT Crop 列

# ---------- 绘图风格 ----------
TITLE_FONTSIZE = 25
TEXT_FONTSIZE = 25
ROI_BOX_COLOR = (255, 0, 0)
ROI_BOX_WIDTH = 3
IMAGE_BORDER_COLOR = "black"
IMAGE_BORDER_WIDTH = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. 基础工具函数
# =========================================================
def read_image_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def calc_bpp(file_size_bytes, image_shape_hw):
    h, w = image_shape_hw
    return file_size_bytes * 8.0 / (h * w)

def calc_psnr_np(gt, rec):
    return float(peak_signal_noise_ratio(gt, rec, data_range=255))

def check_openjpeg_tools():
    if not os.path.isfile(OPJ_COMPRESS):
        raise FileNotFoundError(f"opj_compress.exe not found: {OPJ_COMPRESS}")
    if not os.path.isfile(OPJ_DECOMPRESS):
        raise FileNotFoundError(f"opj_decompress.exe not found: {OPJ_DECOMPRESS}")

def get_box_from_center(center, width, height, img_w, img_h):
    cx, cy = center
    x1 = max(cx - width // 2, 0)
    y1 = max(cy - height // 2, 0)
    x2 = min(x1 + width, img_w)
    y2 = min(y1 + height, img_h)
    return (x1, y1, x2, y2)

def draw_roi_box(img_pil, crop_box, color=(255, 0, 0), width=3):
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(crop_box, outline=color, width=width)
    return img

def crop_np(arr, crop_box):
    x1, y1, x2, y2 = crop_box
    return arr[y1:y2, x1:x2, :]

def add_border(ax, color="black", lw=0.8):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


# =========================================================
# 3. 你的 Net 模型推理（MG-DSC / Ballé2018 共用）
# =========================================================
def pad_to_64(img_tensor):
    _, _, h, w = img_tensor.shape
    h_padded = h if h % 64 == 0 else (h // 64 + 1) * 64
    w_padded = w if w % 64 == 0 else (w // 64 + 1) * 64
    pad_h = h_padded - h
    pad_w = w_padded - w
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return img_tensor, h, w, h_padded, w_padded

def load_syntax_model(ckpt_path, h_padded, w_padded, is_high=False, post_processing=False):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    net = Net(
        train_size=(1, h_padded, w_padded, 3),
        test_size=(1, h_padded, w_padded, 3),
        is_high=is_high,
        post_processing=post_processing
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    net.load_state_dict(state)
    net.eval()
    return net

@torch.no_grad()
def run_syntax_model(img_path, ckpt_path, is_high=False, post_processing=False):
    img_pil = Image.open(img_path).convert("RGB")
    gt_np = np.array(img_pil)
    img_tensor = T.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)

    img_tensor, h, w, h_padded, w_padded = pad_to_64(img_tensor)
    input_tensor = img_tensor * 2.0 - 1.0

    net = load_syntax_model(
        ckpt_path=ckpt_path,
        h_padded=h_padded,
        w_padded=w_padded,
        is_high=is_high,
        post_processing=post_processing
    )

    eval_bpp, _, _ = net(input_tensor, 'test')
    bpp_val = eval_bpp.item()

    z3 = net.a_model(input_tensor)
    z3_syntax = z3[:, :net.M, :, :]
    z3_syntax = net.syntax_model(z3_syntax)
    z3_syntax_rounded = torch.round(z3_syntax)

    z3_content = z3[:, net.M:, :, :]
    z3_content_rounded = torch.round(z3_content)

    x_tilde = net.s_model(z3_content_rounded)
    conv_weights = net.conv_weights_gen(z3_syntax_rounded)
    x_tilde_bf = net.batch_conv(conv_weights, x_tilde)

    if net.post_processing:
        x_tilde = net.HAN(x_tilde_bf)
        conv_weights_han = net.conv_weights_gen_HAN(z3_syntax_rounded)
        x_tilde = net.batch_conv(conv_weights_han, x_tilde)
        x_tilde = net.add_mean(x_tilde)
    else:
        x_tilde = x_tilde_bf

    x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255).round()
    rec_np = x_hat[:, :, :h, :w].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    psnr_val = calc_psnr_np(gt_np, rec_np)

    return {
        "image": rec_np,
        "bpp": float(bpp_val),
        "PSNR": float(psnr_val),
    }


# =========================================================
# 4. JPEG2000 推理
# =========================================================
def run_jpeg2000(img_path, ratio):
    check_openjpeg_tools()

    gt_np = read_image_rgb(img_path)
    h, w = gt_np.shape[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        jp2_path = os.path.join(tmp_dir, "tmp.jp2")
        out_png = os.path.join(tmp_dir, "out.png")

        cmd_enc = [OPJ_COMPRESS, "-i", str(img_path), "-o", jp2_path, "-r", str(ratio)]
        enc_proc = subprocess.run(cmd_enc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if enc_proc.returncode != 0 or (not os.path.exists(jp2_path)):
            raise RuntimeError(f"JPEG2000 编码失败\n{enc_proc.stderr}")

        cmd_dec = [OPJ_DECOMPRESS, "-i", jp2_path, "-o", out_png]
        dec_proc = subprocess.run(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if dec_proc.returncode != 0 or (not os.path.exists(out_png)):
            raise RuntimeError(f"JPEG2000 解码失败\n{dec_proc.stderr}")

        rec_np = np.array(Image.open(out_png).convert("RGB"))
        bpp = calc_bpp(os.path.getsize(jp2_path), (h, w))
        psnr = calc_psnr_np(gt_np, rec_np)

    return {
        "image": rec_np,
        "bpp": float(bpp),
        "PSNR": float(psnr),
    }


# =========================================================
# 5. 统一入口
# =========================================================
def run_model_by_name(model_name, img_path):
    if model_name == "JPEG2000":
        return run_jpeg2000(img_path, JP2_RATIOS[0])

    spec = MODEL_SPECS[model_name]
    return run_syntax_model(
        img_path=img_path,
        ckpt_path=spec["ckpt"],
        is_high=spec.get("is_high", False),
        post_processing=spec.get("post_processing", False),
    )


# =========================================================
# 6. 文本格式
# =========================================================
def metric_text(model_name, item):
    if USE_MANUAL_METRICS:
        bpp = MANUAL_METRICS[model_name]["bpp"]
        psnr = MANUAL_METRICS[model_name]["PSNR"]
    else:
        bpp = item["bpp"]
        psnr = item["PSNR"]

    return f"{bpp:.3f} bpp\nPSNR {psnr:.2f} dB"


# =========================================================
# 7. 绘图
# =========================================================
def make_figure():
    gt_pil = Image.open(IMG_PATH).convert("RGB")
    gt_np = np.array(gt_pil)
    h, w = gt_np.shape[:2]

    crop_box = get_box_from_center(CROP_CENTER, CROP_WIDTH, CROP_HEIGHT, w, h)

    gt_show = draw_roi_box(gt_pil, crop_box, color=ROI_BOX_COLOR, width=ROI_BOX_WIDTH) if DRAW_ROI_ON_GT else gt_pil
    gt_show_np = np.array(gt_show)
    gt_crop_np = crop_np(gt_np, crop_box)

    results = {"GT": {"image": gt_show_np}}
    if SHOW_GT_CROP:
        results["GT_Crop"] = {"image": gt_crop_np}

    for name in ["JPEG2000", "Ballé2018", "MG-DSC"]:
        print(f"Running {name} ...")
        results[name] = run_model_by_name(name, IMG_PATH)

    jpeg_crop = crop_np(results["JPEG2000"]["image"], crop_box)
    balle_crop = crop_np(results["Ballé2018"]["image"], crop_box)
    mgdsc_crop = crop_np(results["MG-DSC"]["image"], crop_box)

    if SHOW_GT_CROP:
        display_items = [
            ("Ground Truth", gt_show_np, None, False),
            ("Ground Truth Crop", gt_crop_np, None, True),
            ("JPEG2000", jpeg_crop, metric_text("JPEG2000", results["JPEG2000"]), True),
            ("Ballé2018", balle_crop, metric_text("Ballé2018", results["Ballé2018"]), True),
            ("MG-DSC", mgdsc_crop, metric_text("MG-DSC", results["MG-DSC"]), True),
        ]
    else:
        display_items = [
            ("Ground Truth", gt_show_np, None, False),
            ("JPEG2000", jpeg_crop, metric_text("JPEG2000", results["JPEG2000"]), True),
            ("Ballé2018", balle_crop, metric_text("Ballé2018", results["Ballé2018"]), True),
            ("MG-DSC", mgdsc_crop, metric_text("MG-DSC", results["MG-DSC"]), True),
        ]

    ncols = len(display_items)
    fig_width = 4.2 * ncols
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 4.6))
    if ncols == 1:
        axes = [axes]

    for ax, (title, img, text, add_img_border) in zip(axes, display_items):
        ax.imshow(img)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        if add_img_border:
            add_border(ax, color=IMAGE_BORDER_COLOR, lw=IMAGE_BORDER_WIDTH)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        if text is not None:
            ax.text(
                0.5, -0.075,
                text,
                transform=ax.transAxes,
                fontsize=TEXT_FONTSIZE,
                ha="center",
                va="top",
                linespacing=1.1
            )

    plt.tight_layout(pad=0.6)
    plt.subplots_adjust(wspace=0.04, bottom=0.16)
    plt.savefig(OUT_FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"✅ Figure saved to: {OUT_FIG_PATH}")


# =========================================================
# 8. 主函数
# =========================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"USE_MANUAL_METRICS = {USE_MANUAL_METRICS}")
    make_figure()