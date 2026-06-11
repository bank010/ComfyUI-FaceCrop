# ComfyUI-FaceCrop

基于 **YOLOv8** 的人脸检测 + **1:1 头像裁剪** ComfyUI 自定义节点，
支持真人 / 动漫 / 自动模式。

## 节点列表

| 节点 | 说明 | 输出 |
| --- | --- | --- |
| **人脸检测裁剪 (FaceDetectCrop)** | 对 `IMAGE` 输入检测人脸并裁成 1:1 头像 | `IMAGE` 裁剪图 / `MASK` 人脸框遮罩 / `FLOAT` 置信度 / `BOOLEAN` 是否检测到 |
| **人脸检测裁剪-URL (FaceDetectCropFromURL)** | 从图片 URL 下载后裁剪（对齐原 `/api/detect`） | `IMAGE` / `FLOAT` / `BOOLEAN` |
| **人脸区域遮罩 (FaceDetectBoxMask)** | 仅输出人脸区域 `MASK`，不裁剪，便于接 inpaint | `MASK` / `FLOAT` / `BOOLEAN` |

## 参数

- `mode`：`auto`（先真人后动漫）/ `real`（真人）/ `anime`（动漫）
- `confidence`：置信度阈值，默认 `0.4`
- `expand`：裁剪框扩充倍率，默认 `2.0`（`1.0` 紧贴人脸，`1.8~2.0` 含头部+肩颈）
- `output_size`：输出方形边长，`0` 表示保持自然裁剪尺寸不缩放
- `fallback_original`：未检测到人脸时是否回退原图（否则输出黑图）

## 安装

1. 将本目录整个复制到 `ComfyUI/custom_nodes/ComfyUI-FaceCrop`。
2. 安装依赖：

```bash
cd ComfyUI/custom_nodes/ComfyUI-FaceCrop
pip install -r requirements.txt
```

3. 准备模型文件，统一放到固定目录 **`ComfyUI/models/facecrop/`**：

   ```
   ComfyUI/models/facecrop/
   ├── yolov8n-face.onnx        # 真人（必需，无自动下载）
   └── yolov8x6_animeface.pt    # 动漫（缺失时自动从 HuggingFace Fuyucchi/yolov8_animeface 下载到此目录）
   ```

4. 重启 ComfyUI，在节点菜单 `FaceCrop` 分类下找到这些节点。

## 说明

- IMAGE 节点支持 batch；当多张裁剪结果尺寸不一致时会统一到 batch 内最大尺寸再堆叠。
- 模型按需懒加载并全局缓存，首次推理会稍慢。
- 真人模型为 `onnx`，需安装 `onnxruntime`（已在 requirements 中）。
