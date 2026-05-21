🌟 SD FaceFusion for Automatic1111 (by RunDiffusion)
==========

> Extension of the upstream FaceFusion project tailored for Automatic1111. Adds YOLO-driven auto-padding masks, extra processors, integrated tabs and postprocessing, and model/output paths aligned with Automatic1111.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
Upstream project: [facefusion/facefusion](https://github.com/facefusion/facefusion)


👀 Preview
--------

<img width="1199" alt="image" src="https://github.com/runnitai/sd_facefusion/assets/1633844/7534bc81-1305-427e-b6e8-1b6e0617397c">


🔥 What’s different from upstream
--------
- **Automatic1111 integration**: Adds a dedicated "RD FaceFusion" tab and an optional postprocessing step in txt2img/img2img. Uses Automatic1111 `models` and output paths.
- **YOLO Auto Padding (custom masking)**:
  - Select any Ultralytics YOLO model to detect objects that intersect with or are near faces and automatically apply padding around the face crop.
  - Works with `.pt` models placed in `models/adetailer` (preferred). If a mask is available, it’s used; otherwise bounding boxes are converted to masks. Configurable confidence and intersection threshold.
  - Replaces the old "custom" mask type with a more robust, detection-driven workflow. Manual on/off markers are still available when Auto Padding is off.
- **Additional/extended processors**: Ships face processors like `Face Swapper`, `Face Enhancer` (with smart enhance), `Expression Restorer` (LivePortrait-based options), `Face Editor`, `Face Debugger`, `Lip Syncer`, `Style Changer`, `Style Transfer`, `Frame Enhancer`, `Frame Colorizer` and more under `facefusion/processors/classes`.
- **Enhanced face detection options**: `yolo_face` (default; `yoloface` jobs alias to this) and `yunet` in addition to upstream detectors.
- **Upstream 3.6.1 merge (fork baseline)**: Hyperswap default swapper, swap weight slider, xseg_1/2/3 occluders, face mask areas, Deep Swapper + Background Remover processors, TensorRT + multi-GPU inference pools, output image/video scale. Content analyser / `nsfw_1` intentionally not ported. Processor layout remains `processors/classes/` (upstream `modules/` migration is planned).
- **Job system and live previews**: Queue jobs, multi-step processing, and preview frames while processing.
- **Model/download alignment**: Auto-downloads models into paths consistent with Automatic1111; leverages onnxruntime-gpu and caches YOLO models in-process for performance.


🧩 Automatic1111 integration
--------
- Tab: Appears as "RD FaceFusion" with File/Live/Benchmark layouts.
- Postprocessing: Optional checkbox to run FaceFusion after generation and swap/enhance using provided source faces.
- Outputs: Written to Automatic1111 temp/output dirs; model paths resolve via `modules.paths_internal.models_path`.


🎯 YOLO Auto Padding (custom object-aware masking)
--------
- UI: In Masking options, select a model in "Auto Padding Model". When a model is selected:
  - "Detection Confidence" (0–1) and "Intersection Threshold (px)" appear.
  - Manual mask timing buttons are hidden (you can re-enable by setting model to "None").
- Models: Place `.pt` files in `models/adetailer/`. The dropdown lists available files from there.
- Behavior: The worker detects objects, filters by confidence, and checks intersection/proximity to each face. If intersecting/near, padding is applied to the crop. Processors (e.g., Swapper/Enhancer/Expression Restorer) consume the recommended padding automatically.
- Notes:
  - We cache YOLO models and suppress Ultralytics logs to keep the UI responsive.
  - Legacy "custom mask" is deprecated in favor of Auto Padding.


🎨 Style processors
--------
- **Style Changer** (face-aware ONNX, per-face + background)
  - Models: dual ONNX models per style (head/background) auto-downloaded to `extensions/sd_facefusion/facefusion/.assets/models/style/`.
    - Available styles include: `anime`, `3d`, `handdrawn`, `sketch`, `artstyle`, `design`, `illustration`.
  - Options: "Selected Style", "Style Target" (`source` or `target`), and "Skip Head" when applying to target.
  - Behavior: detects faces, warps/crops the head region, stylizes head and/or background, and blends using an internal alpha mask.
  - Performance: runs via onnxruntime; GPU providers recommended for speed.

- **Style Transfer** (global neural style transfer, PyTorch)
  - Model: `style_net-TIP-final.pth` auto-downloaded to `extensions/sd_facefusion/facefusion/.assets/models/style/`.
  - Inputs: one or more style images (UI: provide style images; multiple are averaged by default). Works on images and video frames.
  - Behavior: performs sequence-level global feature sharing for video (samples frames at intervals, precomputes norms) to keep style consistent across frames.
  - Performance: CUDA recommended. CPU works but is slower.

UI notes:
- Enable the processors under "Processors". Style Changer options appear in the left panel; Style Transfer options appear near the Source panel (right column) with style image inputs.
- `Style Transfer` is a non-face processor (applies to the whole frame). `Style Changer` is face-aware and participates in face masking options.


🧠 Processors available (high level)
--------
- **Face Swapper**: Primary swapping module with optional Pixel Boost.
- **Face Enhancer**: `gfpgan_1.4` etc., with Smart Enhance controls and minimum-size thresholds.
- **Expression Restorer**: LivePortrait-based expression constraints/limits.
- **Face Editor / Debugger**: Fine-grained attribute controls and visual debugging overlays.
- **Lip Syncer**: Audio-driven lip movement.
- **Style Changer / Style Transfer**, **Frame Enhancer / Frame Colorizer**.

You can enable/disable processors and their options in the UI; arguments are also available via our program interface.


🧭 Differences vs upstream at a glance
--------
- Integrated into Automatic1111 (UI tabs and postprocessing APIs) rather than a standalone app only.
- YOLO-driven Auto Padding pipeline replaces upstream custom mask type and is wired into face selection and processors.
- Uses Automatic1111 model/output paths and supports YOLO models from `models/adetailer`.
- Expanded detectors (`yoloface`, `yunet`) and default detector set to `yoloface`.
- Caching and runtime handling for Ultralytics models; quieter logs; onnxruntime-gpu ensured.
- Some upstream CLI commands still exist, but the typical flow is via the Automatic1111 UI.

Upstream reference: [facefusion/facefusion](https://github.com/facefusion/facefusion)


🛠 Installation
--------
- Install this extension by URL in Automatic1111 like any other extension.
- First run auto-installs requirements and onnxruntime-gpu; torchaudio is installed to match your Torch/CUDA build.
- For YouTube URLs, this fork uses `yt-dlp` (no manual `pytube` patching required).


📁 Model paths
--------
- FaceFusion models are auto-downloaded on demand to the extension’s model folders.
- YOLO models for Auto Padding: drop `.pt` files into `models/adetailer/`.


⚙️ Configuration tips
--------
- `face_detector_model`: default `yoloface`.
- Auto Padding state keys: `auto_padding_model`, `auto_padding_confidence` (default 0.5), `auto_padding_intersection_threshold` (default 50 px).
- Manual mask timing remains available when Auto Padding is disabled (Enable/Disable/Clear markers per frame).


⚠️ Disclaimer
--------
We acknowledge the unethical potential of face manipulation technology and are dedicated to establishing safeguards against misuse. This software abstains from processing inappropriate content such as nudity, graphic content, and other sensitive material.

We do not collaborate with any websites promoting the unauthorized use of this software. Users seeking to engage in such activities may be banned from our community.


📚 Documentation
--------
- Upstream docs: `https://docs.facefusion.io`  
- This fork’s UI is self-documenting via tooltips and labels; key differences are summarized above.
