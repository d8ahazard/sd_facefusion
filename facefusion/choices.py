import logging
from typing import List, Optional, Sequence

from facefusion.common_helper import create_float_range, create_int_range
from facefusion.typing import Angle, ExecutionProviderSet, FaceDetectorSet, FaceLandmarkerModel, FaceMaskArea, \
    FaceMaskAreaSet, FaceMaskRegion, FaceMaskType, FaceSelectorMode, FaceSelectorOrder, Gender, JobStatus, \
    LogLevelSet, OutputAudioEncoder, OutputVideoEncoder, OutputVideoPreset, Race, Score, TempFrameFormat, \
    UiWorkflow, VideoMemoryStrategy

video_memory_strategies: List[VideoMemoryStrategy] = ['strict', 'moderate', 'tolerant']

face_detector_set: FaceDetectorSet = \
    {
        'many': ['640x640'],
        'retinaface': ['160x160', '320x320', '480x480', '512x512', '640x640'],
        'scrfd': ['160x160', '320x320', '480x480', '512x512', '640x640'],
        'yolo_face': ['640x640'],
        'yoloface': ['640x640'],
        'yunet': ['640x640'],
    }
face_mask_area_set: FaceMaskAreaSet = \
    {
        'upper-face': [0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17],
        'lower-face': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31],
        'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
    }
face_mask_areas: List[FaceMaskArea] = list(face_mask_area_set.keys())
auto_padding_mask_areas_default: List[FaceMaskArea] = ['upper-face', 'lower-face', 'mouth']


def normalize_face_mask_areas(face_mask_areas) -> List[FaceMaskArea]:
    """Flatten CheckboxGroup / ini / JSON values into a list of valid area names."""
    if face_mask_areas is None:
        return []
    if isinstance(face_mask_areas, str):
        face_mask_areas = [part.strip() for part in face_mask_areas.replace(',', ' ').split() if part.strip()]
    flat: List[FaceMaskArea] = []
    for item in face_mask_areas:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, str) and sub in face_mask_area_set:
                    flat.append(sub)
        elif isinstance(item, str) and item in face_mask_area_set:
            flat.append(item)
    return flat


def resolve_face_mask_areas(face_mask_areas: Optional[List[FaceMaskArea]]) -> List[FaceMaskArea]:
    """Manual area mask: empty selection means all areas (upstream FaceFusion default)."""
    normalized = normalize_face_mask_areas(face_mask_areas)
    if normalized:
        return normalized
    return list(face_mask_area_set.keys())


def resolve_auto_padding_mask_areas(auto_padding_mask_areas) -> List[FaceMaskArea]:
    """YOLO auto exclusions: empty list is valid (disables near-face area exclusions). None uses default trio."""
    if auto_padding_mask_areas is None:
        return list(auto_padding_mask_areas_default)
    return normalize_face_mask_areas(auto_padding_mask_areas)


def auto_padding_mask_areas_for_ui(stored) -> List[FaceMaskArea]:
    """CheckboxGroup value: preserve explicit [], default only when unset."""
    if stored is None:
        return list(auto_padding_mask_areas_default)
    return normalize_face_mask_areas(stored)


def auto_padding_mask_areas_for_runtime(stored) -> List[FaceMaskArea]:
    """Processing: same semantics as auto_padding_mask_areas_for_ui."""
    return auto_padding_mask_areas_for_ui(stored)


face_landmarker_models: List[FaceLandmarkerModel] = ['many', '2dfan4', 'peppa_wutz']
face_selector_modes: List[FaceSelectorMode] = ['one', 'reference']
face_selector_orders: List[FaceSelectorOrder] = ['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large',
                                                 'large-small', 'best-worst', 'worst-best']
face_selector_genders: List[Gender] = ['female', 'male']
face_selector_races: List[Race] = ['white', 'black', 'latino', 'asian', 'indian', 'arabic']
face_mask_types: List[FaceMaskType] = ['box', 'occlusion', 'area', 'region', 'custom']
face_mask_regions: List[FaceMaskRegion] = ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses',
                                           'nose', 'mouth', 'upper-lip', 'lower-lip']
temp_frame_formats: List[TempFrameFormat] = ['bmp', 'jpg', 'png']
output_audio_encoders: List[OutputAudioEncoder] = ['aac', 'libmp3lame', 'libopus', 'libvorbis']
output_video_encoders: List[OutputVideoEncoder] = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc',
                                                   'h264_amf', 'hevc_amf', 'h264_videotoolbox', 'hevc_videotoolbox']
output_video_presets: List[OutputVideoPreset] = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium',
                                                 'slow', 'slower', 'veryslow']

image_template_sizes: List[float] = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4]
video_template_sizes: List[int] = [240, 360, 480, 540, 720, 1080, 1440, 2160, 4320]

log_level_set: LogLevelSet = \
    {
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

execution_provider_set: ExecutionProviderSet = \
    {
        'cpu': 'CPUExecutionProvider',
        'coreml': 'CoreMLExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'directml': 'DmlExecutionProvider',
        'openvino': 'OpenVINOExecutionProvider',
        'rocm': 'ROCMExecutionProvider',
        'tensorrt': 'TensorrtExecutionProvider',
    }

ui_workflows: List[UiWorkflow] = ['instant_runner', 'job_runner', 'job_manager']
job_statuses: List[JobStatus] = ['drafted', 'queued', 'completed', 'failed']

execution_thread_count_range: Sequence[int] = create_int_range(1, 32, 1)
execution_queue_count_range: Sequence[int] = create_int_range(1, 4, 1)
system_memory_limit_range: Sequence[int] = create_int_range(0, 128, 4)
face_detector_angles: Sequence[Angle] = create_int_range(0, 270, 90)
face_detector_score_range: Sequence[Score] = create_float_range(0.0, 1.0, 0.05)
face_landmarker_score_range: Sequence[Score] = create_float_range(0.0, 1.0, 0.05)
face_mask_blur_range: Sequence[float] = create_float_range(0.0, 1.0, 0.05)
face_mask_padding_range: Sequence[int] = create_int_range(0, 100, 1)
face_selector_age_range: Sequence[int] = create_int_range(0, 100, 1)
reference_face_distance_range: Sequence[float] = create_float_range(0.0, 1.5, 0.05)
output_image_quality_range: Sequence[int] = create_int_range(0, 100, 1)
output_video_quality_range: Sequence[int] = create_int_range(0, 100, 1)
output_image_scale_range: Sequence[float] = create_float_range(0.25, 8.0, 0.25)
output_video_scale_range: Sequence[float] = create_float_range(0.25, 8.0, 0.25)
download_scopes: List[str] = ['lite', 'full']
