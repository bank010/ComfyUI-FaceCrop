"""
人脸检测与裁剪核心逻辑（ComfyUI 独立版）

与原 FastAPI 模块的检测策略保持一致：
- 真人模型：yolov8n-face.onnx
- 动漫模型：yolov8x6_animeface.pt（缺失时自动从 HuggingFace 下载）
- auto 模式优先真人，未命中再尝试动漫

不依赖 FastAPI / app.settings，可在纯 ComfyUI 环境运行。
"""
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("ComfyUI-FaceCrop")

PLUGIN_DIR = Path(__file__).parent
REAL_MODEL_FILENAME = "yolov8n-face.onnx"
ANIME_MODEL_FILENAME = "yolov8x6_animeface.pt"
ANIME_HF_REPO = "Fuyucchi/yolov8_animeface"

_real_model = None
_anime_model = None


def get_model_dir() -> Path:
    """固定模型目录：ComfyUI/models/facecrop/

    若不在 ComfyUI 环境（找不到 folder_paths），回退到插件目录下的
    models/facecrop/，保证脱离 ComfyUI 时仍可运行。
    """
    try:
        import folder_paths  # type: ignore

        base = Path(folder_paths.models_dir)
    except Exception:
        base = PLUGIN_DIR / "models"
    return base / "facecrop"


def _find_model(filename: str) -> Path | None:
    p = get_model_dir() / filename
    return p if p.exists() else None


def _download_anime_model() -> Path:
    """从 HuggingFace 下载动漫人脸模型到固定模型目录。"""
    from huggingface_hub import hf_hub_download

    target_dir = get_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    log.info("正在从 HuggingFace 下载动漫人脸模型: %s", ANIME_HF_REPO)
    path = hf_hub_download(
        repo_id=ANIME_HF_REPO,
        filename=ANIME_MODEL_FILENAME,
        local_dir=str(target_dir),
    )
    log.info("动漫人脸模型已保存至: %s", path)
    return Path(path)


def get_real_model():
    from ultralytics import YOLO

    global _real_model
    if _real_model is None:
        model_path = _find_model(REAL_MODEL_FILENAME)
        if model_path is None:
            raise FileNotFoundError(
                f"未找到真人模型 {REAL_MODEL_FILENAME}，请放入: {get_model_dir()}"
            )
        _real_model = YOLO(str(model_path), task="detect")
    return _real_model


def get_anime_model():
    from ultralytics import YOLO

    global _anime_model
    if _anime_model is None:
        model_path = _find_model(ANIME_MODEL_FILENAME)
        if model_path is None:
            model_path = _download_anime_model()
        _anime_model = YOLO(str(model_path), task="detect")
    return _anime_model


def crop_square(
    img: Image.Image,
    box: tuple[int, int, int, int],
    expand: float = 2.0,
) -> Image.Image:
    """将检测框扩展为 1:1 正方形后裁剪，居中于原始检测框。

    expand 控制扩充倍率：
        1.0 = 紧贴检测框
        1.2 = 略微扩充（紧凑人脸）
        1.8~2.0 = 包含完整头部 + 部分肩颈

    当人脸贴近图像边缘、expand 较大时，期望的正方形可能超出图像范围。
    此时不再裁出长方形再强行 resize（会导致画面被拉伸/压缩），而是：
    1) 将边长限制为不超过原图较短边，保证正方形一定能放进图内；
    2) 平移裁剪窗口使其完整落在图像内。
    这样始终裁出真正的 1:1，不变形、无黑边；代价是边缘场景下实际倍率会
    略小于设定的 expand（取图像能容纳的最大正方形）。
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    img_w, img_h = img.size

    side = math.ceil(max(w, h) * expand)
    side = max(1, min(side, img_w, img_h))

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    offset_y = int((expand - 1.0) * h * 0.1)
    cy -= offset_y

    half = side // 2
    left = cx - half
    top = cy - half
    # 平移窗口，使其完整落在图像内（side<=两边，必定可行）
    left = min(max(left, 0), img_w - side)
    top = min(max(top, 0), img_h - side)

    return img.crop((left, top, left + side, top + side))


def _run_detect(model, img_arr: np.ndarray, conf: float):
    """运行单个模型检测，返回 (best_box_xyxy, best_conf) 或 None。"""
    results = model.predict(source=img_arr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best_idx = int(boxes.conf.argmax())
    xyxy = boxes.xyxy[best_idx].int().tolist()
    best_conf = float(boxes.conf[best_idx])
    return xyxy, best_conf


def detect_box(img: Image.Image, conf: float, mode: str):
    """对 PIL 图像执行人脸检测，返回 ((x1,y1,x2,y2), conf) 或 None。"""
    img_arr = np.array(img.convert("RGB"))
    detection = None

    if mode in ("real", "auto"):
        try:
            detection = _run_detect(get_real_model(), img_arr, conf)
        except FileNotFoundError as e:
            log.warning("真人模型加载失败: %s", e)

    if detection is None and mode in ("anime", "auto"):
        try:
            detection = _run_detect(get_anime_model(), img_arr, conf)
        except FileNotFoundError as e:
            log.warning("动漫模型加载失败: %s", e)

    return detection


def detect_box_adaptive(
    img: Image.Image,
    mode: str,
    start: float = 0.8,
    floor: float = 0.2,
    step: float = 0.1,
):
    """自适应阈值检测。

    从 start 阈值开始检测，未命中则按 step 逐步降低，直到 floor。
    一旦命中立即返回 ((x1,y1,x2,y2), conf)，全程未命中返回 None。
    """
    conf = start
    while conf >= floor - 1e-9:
        c = round(conf, 4)
        detection = detect_box(img, c, mode)
        if detection is not None:
            log.info("自适应检测命中: 阈值=%.2f, conf=%.4f", c, detection[1])
            return detection
        log.info("自适应检测未命中: 阈值=%.2f，继续降低", c)
        conf -= step
    return None


def detect_and_crop_pil(
    img: Image.Image,
    *,
    conf: float = 0.4,
    expand: float = 2.0,
    mode: str = "auto",
    size: int | None = None,
):
    """对 PIL 图像执行检测+1:1 裁剪。返回 (cropped_image|None, box|None, conf)。"""
    img = img.convert("RGB")
    detection = detect_box(img, conf, mode)
    if detection is None:
        return None, None, 0.0

    box, best_conf = detection
    log.info("检测到人脸: conf=%.4f, box=%s", best_conf, box)
    cropped = crop_square(img, tuple(box), expand=expand)
    if size and size > 0:
        cropped = cropped.resize((size, size), Image.LANCZOS)
    return cropped, box, best_conf


def detect_and_crop_pil_adaptive(
    img: Image.Image,
    *,
    expand: float = 2.0,
    mode: str = "auto",
    size: int | None = None,
    start: float = 0.8,
    floor: float = 0.2,
    step: float = 0.1,
):
    """自适应阈值检测 + 1:1 裁剪。返回 (cropped|None, box|None, conf)。

    从 start 阈值开始，未检出则逐步降到 floor；全程未检出返回 (None, None, 0.0)。
    """
    img = img.convert("RGB")
    detection = detect_box_adaptive(img, mode, start=start, floor=floor, step=step)
    if detection is None:
        return None, None, 0.0

    box, best_conf = detection
    cropped = crop_square(img, tuple(box), expand=expand)
    if size and size > 0:
        cropped = cropped.resize((size, size), Image.LANCZOS)
    return cropped, box, best_conf
