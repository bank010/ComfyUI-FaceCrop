"""
ComfyUI 节点定义

提供三个节点：
- FaceDetectCrop        : 对 IMAGE 输入做人脸检测 + 1:1 裁剪
- FaceDetectCropFromURL : 从 URL 下载图片后检测 + 裁剪（对齐原 API 行为）
- FaceDetectBoxMask     : 仅输出人脸区域 MASK，不裁剪（便于后续 inpaint 等）
"""
import io
import logging
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from .face_core import (
    crop_square,
    detect_and_crop_pil,
    detect_and_crop_pil_adaptive,
    detect_box,
)

log = logging.getLogger("ComfyUI-FaceCrop")

MODES = ["auto", "real", "anime"]

# 检测置信度阈值（固定，不在节点上暴露为可调参数）
# 检测从 DEFAULT_CONFIDENCE 开始，未命中按 CONFIDENCE_STEP 逐步降低，
# 直到 MIN_CONFIDENCE；仍未命中则报错。
DEFAULT_CONFIDENCE = 0.8
MIN_CONFIDENCE = 0.2
CONFIDENCE_STEP = 0.1


# ------------------------- 张量 <-> PIL 转换 -------------------------

def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE 张量 [H,W,C] (0-1 float) -> PIL.Image (RGB)。"""
    arr = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL.Image -> ComfyUI IMAGE 张量 [1,H,W,C] (0-1 float)。"""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _stack_same_size(images: list[Image.Image]) -> torch.Tensor:
    """将多张可能不同尺寸的 PIL 图统一尺寸后堆叠为一个 batch 张量。"""
    if not images:
        # 返回 1x1 黑图，避免下游崩溃
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    target = max(im.width for im in images), max(im.height for im in images)
    tensors = []
    for im in images:
        if im.size != target:
            im = im.resize(target, Image.LANCZOS)
        tensors.append(pil_to_tensor(im))
    return torch.cat(tensors, dim=0)


# ------------------------------ 节点 1 ------------------------------

class FaceDetectCrop:
    """人脸检测 + 1:1 头像裁剪。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (MODES, {"default": "auto"}),
                "expand": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1},
                ),
                "output_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("cropped_image", "face_mask", "confidence", "detected")
    FUNCTION = "run"
    CATEGORY = "FaceCrop"

    def run(self, image, mode, expand, output_size):
        size = output_size if output_size > 0 else None
        cropped_list: list[Image.Image] = []
        mask_list: list[torch.Tensor] = []
        confs: list[float] = []

        for i in range(image.shape[0]):
            pil = tensor_to_pil(image[i])
            # 从 DEFAULT_CONFIDENCE 开始，未命中自动降低阈值重试，直到 MIN_CONFIDENCE
            cropped, box, conf = detect_and_crop_pil_adaptive(
                pil,
                expand=expand,
                mode=mode,
                size=size,
                start=DEFAULT_CONFIDENCE,
                floor=MIN_CONFIDENCE,
                step=CONFIDENCE_STEP,
            )

            if cropped is None:
                # 降到 MIN_CONFIDENCE 仍未检出：直接报错，不返回黑图/原图
                raise RuntimeError(
                    f"未检测到人脸：阈值已从 {DEFAULT_CONFIDENCE} 逐步降到 "
                    f"{MIN_CONFIDENCE} 仍无结果（batch 第 {i} 张图，mode={mode}）。"
                )

            mask = torch.zeros((pil.height, pil.width), dtype=torch.float32)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1.0
            confs.append(conf)

            cropped_list.append(cropped)
            mask_list.append(mask)

        out_image = _stack_same_size(cropped_list)

        # MASK 对齐原图尺寸（按 batch 内最大尺寸统一）
        mh = max(m.shape[0] for m in mask_list)
        mw = max(m.shape[1] for m in mask_list)
        padded = []
        for m in mask_list:
            if m.shape != (mh, mw):
                pm = torch.zeros((mh, mw), dtype=torch.float32)
                pm[: m.shape[0], : m.shape[1]] = m
                m = pm
            padded.append(m)
        out_mask = torch.stack(padded, dim=0)

        avg_conf = float(np.mean(confs)) if confs else 0.0
        # 走到这里说明每张图都已成功检出（否则上面已报错）
        return (out_image, out_mask, avg_conf, True)


# ------------------------------ 节点 2 ------------------------------

class FaceDetectCropFromURL:
    """从图片 URL 下载后执行人脸检测 + 1:1 裁剪（对齐原 /api/detect）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "mode": (MODES, {"default": "auto"}),
                "expand": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1},
                ),
                "output_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 8},
                ),
            },
            "optional": {
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("cropped_image", "confidence", "detected")
    FUNCTION = "run"
    CATEGORY = "FaceCrop"

    def _download(self, url: str, proxy_url: str) -> Image.Image:
        import httpx

        url = url.strip()
        if not urlparse(url).scheme:
            url = "https://" + url

        client_kwargs = dict(
            timeout=20,
            follow_redirects=True,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if proxy_url.strip():
            client_kwargs["proxy"] = proxy_url.strip()

        with httpx.Client(**client_kwargs) as client:
            resp = client.get(url)
            resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def run(self, url, mode, expand, output_size, proxy_url=""):
        size = output_size if output_size > 0 else None
        try:
            pil = self._download(url, proxy_url)
        except Exception as e:
            log.warning("图片下载失败: %s — %s", url, e)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), 0.0, False)

        cropped, _box, conf = detect_and_crop_pil(
            pil, conf=DEFAULT_CONFIDENCE, expand=expand, mode=mode, size=size
        )
        if cropped is None:
            log.info("未检测到人脸: %s", url)
            fallback = pil if size is None else pil.resize((size, size), Image.LANCZOS)
            return (pil_to_tensor(fallback), 0.0, False)

        return (pil_to_tensor(cropped), conf, True)


# ------------------------------ 节点 3 ------------------------------

class FaceDetectBoxMask:
    """仅检测人脸并输出区域 MASK（不裁剪），方便接 inpaint / 合成。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (MODES, {"default": "auto"}),
                "expand": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("face_mask", "confidence", "detected")
    FUNCTION = "run"
    CATEGORY = "FaceCrop"

    def run(self, image, mode, expand):
        mask_list: list[torch.Tensor] = []
        confs: list[float] = []
        any_detected = False

        for i in range(image.shape[0]):
            pil = tensor_to_pil(image[i])
            mask = torch.zeros((pil.height, pil.width), dtype=torch.float32)
            detection = detect_box(pil, DEFAULT_CONFIDENCE, mode)
            if detection is not None:
                any_detected = True
                (x1, y1, x2, y2), conf = detection
                if expand > 1.0:
                    w, h = x2 - x1, y2 - y1
                    dx, dy = int(w * (expand - 1) / 2), int(h * (expand - 1) / 2)
                    x1, y1 = max(x1 - dx, 0), max(y1 - dy, 0)
                    x2 = min(x2 + dx, pil.width)
                    y2 = min(y2 + dy, pil.height)
                mask[y1:y2, x1:x2] = 1.0
                confs.append(conf)
            else:
                confs.append(0.0)
            mask_list.append(mask)

        mh = max(m.shape[0] for m in mask_list)
        mw = max(m.shape[1] for m in mask_list)
        padded = []
        for m in mask_list:
            if m.shape != (mh, mw):
                pm = torch.zeros((mh, mw), dtype=torch.float32)
                pm[: m.shape[0], : m.shape[1]] = m
                m = pm
            padded.append(m)
        out_mask = torch.stack(padded, dim=0)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return (out_mask, avg_conf, any_detected)


NODE_CLASS_MAPPINGS = {
    "FaceDetectCrop": FaceDetectCrop,
    "FaceDetectCropFromURL": FaceDetectCropFromURL,
    "FaceDetectBoxMask": FaceDetectBoxMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectCrop": "人脸检测裁剪 (Face Detect & Crop)",
    "FaceDetectCropFromURL": "人脸检测裁剪-URL (Face Crop from URL)",
    "FaceDetectBoxMask": "人脸区域遮罩 (Face Box Mask)",
}
