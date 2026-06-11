"""
ComfyUI Face Crop 插件

基于 YOLOv8 的人脸检测 + 1:1 头像裁剪自定义节点。
支持真人 / 动漫 / 自动三种检测模式。

将本目录整个复制到 ComfyUI/custom_nodes/ 下即可使用。
"""
import importlib.util
import os
import sys

# 某些 ComfyUI fork(如 autodl 的 ComfyUI_New)加载自定义节点时，会把模块名
# 设成完整绝对路径且不注册子模块搜索路径，导致 `from .nodes import ...` 这类
# 相对导入失败。这里改为按文件路径显式加载，并注册成一个固定的合成包名，
# 使 nodes.py 内部的 `from .face_core import ...` 也能稳定工作。
_PKG = "comfyui_facecrop"
_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_package():
    """在 sys.modules 中创建一个带子模块搜索路径的合成包。"""
    if _PKG in sys.modules:
        return sys.modules[_PKG]
    spec = importlib.util.spec_from_file_location(
        _PKG,
        os.path.join(_DIR, "__init__.py"),
        submodule_search_locations=[_DIR],
    )
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [_DIR]
    sys.modules[_PKG] = pkg
    return pkg


def _load_submodule(name):
    full = f"{_PKG}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, os.path.join(_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    # 优先走标准相对导入（标准版 ComfyUI 走这里）。
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    # 回退：手动按文件路径加载（兼容把模块名设成绝对路径的 fork）。
    _ensure_package()
    _load_submodule("face_core")
    _nodes = _load_submodule("nodes")
    NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
