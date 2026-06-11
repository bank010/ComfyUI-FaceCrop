"""
ComfyUI Face Crop 插件

基于 YOLOv8 的人脸检测 + 1:1 头像裁剪自定义节点。
支持真人 / 动漫 / 自动三种检测模式。

将本目录整个复制到 ComfyUI/custom_nodes/ 下即可使用。
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
