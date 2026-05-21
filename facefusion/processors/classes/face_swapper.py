import os
from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy

from facefusion import config, inference_manager, process_manager, state_manager, wording
from facefusion import logger
from facefusion.common_helper import get_first
from facefusion.execution import has_execution_provider
from facefusion.face_analyser import get_many_faces, get_one_face, get_average_faces
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import has_image, in_directory, is_image, is_video, \
    resolve_relative_path, same_file_extension
from facefusion.inference_manager import get_static_model_initializer
from facefusion.jobs import job_store
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from facefusion.processors.typing import FaceSwapperInputs
from facefusion.program_helper import find_argument_group, suggest_face_swapper_pixel_boost_choices
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import ApplyStateItem, Args, Embedding, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, \
    QueuePayload, VisionFrame, Padding
from facefusion.vision import read_image, read_static_image, unpack_resolution, write_image
from facefusion.workers.classes.face_masker import FaceMasker, clear_yolo_model_cache
from facefusion.workers.core import clear_worker_modules


def update_padding(padding: Padding, frame_number: int) -> Padding:
    if frame_number == -1:
        return padding

    disabled_times = state_manager.get_item('mask_disabled_times') or []
    enabled_times = state_manager.get_item('mask_enabled_times') or []

    latest_disabled_frame = max([frame for frame in disabled_times if frame <= frame_number], default=None)
    latest_enabled_frame = max([frame for frame in enabled_times if frame <= frame_number], default=None)
    
    # Padding is disabled by default
    # Only enable padding if there's an enabled event that's more recent than any disabled event
    if latest_enabled_frame is not None and (
            latest_disabled_frame is None or latest_enabled_frame > latest_disabled_frame):
        return padding  # Enable padding
    
    # Default: disable padding
    new_padding = (0, 0, 0, 0)
    return new_padding


class FaceSwapper(BaseProcessor):
    MODEL_SET: ModelSet = \
        {
            'blendswap_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/blendswap_256.hash',
                                    'path': resolve_relative_path('../.assets/models/blendswap_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/blendswap_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/blendswap_256.onnx')
                                }
                        },
                    'type': 'blendswap',
                    'template': 'ffhq_512',
                    'size': (256, 256),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'ghost_1_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_1_256.hash',
                                    'path': resolve_relative_path('../.assets/models/ghost_1_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.hash',
                                    'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_1_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/ghost_1_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.onnx',
                                    'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
                                }
                        },
                    'type': 'ghost',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'ghost_2_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_2_256.hash',
                                    'path': resolve_relative_path('../.assets/models/ghost_2_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.hash',
                                    'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_2_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/ghost_2_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.onnx',
                                    'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
                                }
                        },
                    'type': 'ghost',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'ghost_3_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_3_256.hash',
                                    'path': resolve_relative_path('../.assets/models/ghost_3_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.hash',
                                    'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_3_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/ghost_3_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.onnx',
                                    'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
                                }
                        },
                    'type': 'ghost',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'hififace_unofficial_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/hififace_unofficial_256.hash',
                                    'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_hififace.hash',
                                    'path': resolve_relative_path('../.assets/models/crossface_hififace.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/hififace_unofficial_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_hififace.onnx',
                                    'path': resolve_relative_path('../.assets/models/crossface_hififace.onnx')
                                }
                        },
                    'type': 'hififace',
                    'template': 'mtcnn_512',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'hyperswap_1a_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.hash',
                                    'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.onnx')
                                }
                        },
                    'type': 'hyperswap',
                    'template': 'arcface_128',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'hyperswap_1b_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.hash',
                                    'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.onnx')
                                }
                        },
                    'type': 'hyperswap',
                    'template': 'arcface_128',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'hyperswap_1c_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1c_256.hash',
                                    'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1c_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.onnx')
                                }
                        },
                    'type': 'hyperswap',
                    'template': 'arcface_128',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                },
            'inswapper_128':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.hash',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128.onnx')
                                }
                        },
                    'type': 'inswapper',
                    'template': 'arcface_128_v2',
                    'size': (128, 128),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'inswapper_128_fp16':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.hash',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx',
                                    'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx')
                                }
                        },
                    'type': 'inswapper',
                    'template': 'arcface_128_v2',
                    'size': (128, 128),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'simswap_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.hash',
                                    'path': resolve_relative_path('../.assets/models/simswap_256.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_simswap.hash',
                                    'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/simswap_256.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_simswap.onnx',
                                    'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
                                }
                        },
                    'type': 'simswap',
                    'template': 'arcface_112_v1',
                    'size': (256, 256),
                    'mean': [0.485, 0.456, 0.406],
                    'standard_deviation': [0.229, 0.224, 0.225]
                },
            'simswap_unofficial_512':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_unofficial_512.hash',
                                    'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.hash')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_simswap.hash',
                                    'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_unofficial_512.onnx',
                                    'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.onnx')
                                },
                            'embedding_converter':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_simswap.onnx',
                                    'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
                                }
                        },
                    'type': 'simswap',
                    'template': 'arcface_112_v1',
                    'size': (512, 512),
                    'mean': [0.0, 0.0, 0.0],
                    'standard_deviation': [1.0, 1.0, 1.0]
                },
            'uniface_256':
                {
                    'hashes':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/uniface_256.hash',
                                    'path': resolve_relative_path('../.assets/models/uniface_256.hash')
                                }
                        },
                    'sources':
                        {
                            'face_swapper':
                                {
                                    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/uniface_256.onnx',
                                    'path': resolve_relative_path('../.assets/models/uniface_256.onnx')
                                }
                        },
                    'type': 'uniface',
                    'template': 'ffhq_512',
                    'size': (256, 256),
                    'mean': [0.5, 0.5, 0.5],
                    'standard_deviation': [0.5, 0.5, 0.5]
                }
        }

    model_key: str = 'face_swapper_model'
    priority: int = 0
    preload: bool = True
    preferred_provider = 'cuda'
    src_cache = {}
    
    # Cached model options to avoid repeated lookups
    _cached_model_options = None
    _cached_model_key = None
    _cached_input_names = None

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, 'processors')
        if group_processors:
            group_processors.add_argument('--face-swapper-model', help=wording.get('help.face_swapper_model'),
                                          default=config.get_str_value('processors.face_swapper_model',
                                                                       'hyperswap_1a_256'),
                                          choices=self.list_models())
            face_swapper_pixel_boost_choices = suggest_face_swapper_pixel_boost_choices(program)
            group_processors.add_argument('--face-swapper-pixel-boost',
                                          help=wording.get('help.face_swapper_pixel_boost'),
                                          default=config.get_str_value('processors.face_swapper_pixel_boost',
                                                                       get_first(face_swapper_pixel_boost_choices)),
                                          choices=face_swapper_pixel_boost_choices)
            from facefusion.processors import choices as processors_choices
            group_processors.add_argument('--face-swapper-weight', help=wording.get('help.face_swapper_weight'),
                                          type=float,
                                          default=config.get_float_value('processors.face_swapper_weight', '0.5'),
                                          choices=processors_choices.face_swapper_weight_range,
                                          metavar='FACE_SWAPPER_WEIGHT')
            job_store.register_step_keys(['face_swapper_model', 'face_swapper_pixel_boost', 'face_swapper_weight'])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item('face_swapper_model', args.get('face_swapper_model'))
        apply_state_item('face_swapper_pixel_boost', args.get('face_swapper_pixel_boost'))
        apply_state_item('face_swapper_weight', args.get('face_swapper_weight'))

    def pre_process(self, mode: ProcessMode) -> bool:
        self.src_cache = {}
        source_paths = state_manager.get_item('source_paths')
        source_paths_2 = state_manager.get_item('source_paths_2')
        if not has_image(source_paths) and not has_image(source_paths_2):
            logger.error(wording.get('choose_image_source') + wording.get('exclamation_mark'), __name__)
            return False
        source_faces = get_average_faces()
        source_face_values = [value for value in source_faces.values()]
        target_folder = state_manager.get_item('target_folder')
        is_batch = False
        if target_folder is not None and target_folder != "" and os.path.isdir(target_folder):
            is_batch = True
            logger.info("Batch processing is enabled", __name__)
        if not len(source_face_values):
            logger.error(wording.get('no_source_face_detected') + wording.get('exclamation_mark'), __name__)
            return False
        if mode in ['output', 'preview'] and not is_image(state_manager.get_item('target_path')) and not is_video(
                state_manager.get_item('target_path')) and not is_batch:
            logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not in_directory(state_manager.get_item('output_path')) and not is_batch:
            logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not is_batch and not same_file_extension(
                [state_manager.get_item('target_path'), state_manager.get_item('output_path')]):
            logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
            return False
        return True

    def post_process(self) -> None:
        self.src_cache = {}
        self.clear_model_cache()  # Clear cached model options
        if state_manager.get_item("video_memory_strategy") in ["strict", "moderate"]:
            self.clear_inference_pool()
            clear_yolo_model_cache()  # Clear YOLO model cache to free memory
        if state_manager.get_item("video_memory_strategy") == "strict":
            clear_worker_modules()

    def process_frame(self, inputs: FaceSwapperInputs) -> VisionFrame:
        reference_faces = inputs.get('reference_faces') or {}
        source_faces = inputs.get('source_faces') or {}
        face_selector_mode = state_manager.get_item('face_selector_mode')
        source_face = next(iter(source_faces.values()), None) if face_selector_mode != 'reference' else None
        target_frame_number = inputs.get('target_frame_number', -1)
        target_vision_frame = inputs.get('target_vision_frame')
        
        # Check for cached faces from face buffer
        cached_faces = inputs.get('cached_faces')
        if cached_faces is not None:
            many_faces = cached_faces
        else:
            many_faces = sort_and_filter_faces(
                get_many_faces([target_vision_frame], is_target_frame=True),
                vision_frame=target_vision_frame,
            )
        
        # Collect settings once per frame to avoid repeated state lookups
        auto_padding_model = state_manager.get_item('auto_padding_model')
        padding = state_manager.get_item('face_mask_padding') or (0, 0, 0, 0)
        
        if not auto_padding_model or auto_padding_model == "None":
            padding = update_padding(padding, target_frame_number) or (0, 0, 0, 0)
        else:
            padding = (0, 0, 0, 0)
        
        # Cache settings for swap_face calls
        cached_settings = {
            'pixel_boost_size': unpack_resolution(state_manager.get_item('face_swapper_pixel_boost')),
            'face_mask_types': state_manager.get_item('face_mask_types') or [],
            'face_mask_blur': state_manager.get_item('face_mask_blur') or 0.3,
            'face_mask_regions': state_manager.get_item('face_mask_regions'),
            'face_mask_areas': state_manager.get_item('face_mask_areas'),
            'auto_padding_model': auto_padding_model,
        }

        masker = FaceMasker()
        
        src_idx = 0
        if face_selector_mode == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.swap_face(source_face, target_face, target_vision_frame, src_idx, target_frame_number, padding, cached_settings, masker)
            else:
                logger.info("No target face found", __name__)
        if face_selector_mode == 'reference':
            reference_face_keys = set(reference_faces.keys()) if reference_faces else set()
            source_face_keys = set(source_faces.keys()) if source_faces else set()
            all_keys = reference_face_keys.union(source_face_keys)
            
            reference_face_distance = state_manager.get_item('reference_face_distance')

            for src_face_idx in all_keys:
                ref_faces = reference_faces.get(src_face_idx)
                src_face = source_faces.get(src_face_idx)

                if not ref_faces or not src_face:
                    continue

                similar_faces = find_similar_faces(many_faces, ref_faces, reference_face_distance)

                if similar_faces:
                    for similar_face in similar_faces:
                        target_vision_frame = self.swap_face(
                            src_face, similar_face, target_vision_frame, src_face_idx, target_frame_number, padding, cached_settings, masker
                        )

        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            target_frame_number = queue_payload['frame_number']
            source_faces = queue_payload['source_faces']
            reference_faces = queue_payload['reference_faces']
            target_vision_frame = read_image(target_vision_path)
            
            # Build input dict with cached data if available
            process_inputs = {
                'reference_faces': reference_faces,
                'source_faces': source_faces,
                'target_vision_frame': target_vision_frame,
                'target_frame_number': target_frame_number
            }
            
            # Add cached data from face buffer if available
            if 'cached_faces' in queue_payload:
                process_inputs['cached_faces'] = queue_payload['cached_faces']
            if 'cached_yolo_detections' in queue_payload:
                process_inputs['cached_yolo_detections'] = queue_payload['cached_yolo_detections']
            if 'cached_interpolated' in queue_payload:
                process_inputs['cached_interpolated'] = queue_payload['cached_interpolated']
            
            result_frame = self.process_frame(process_inputs)
            write_image(target_vision_path, result_frame)
            output_frames.append((target_frame_number, target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str, reference_faces=None) -> None:
        if reference_faces is None:
            reference_faces = (
                get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (None, None))
        source_faces = get_average_faces()
        target_vision_frame = read_static_image(target_path)
        result_frame = self.process_frame(
            {
                'reference_faces': reference_faces,
                'source_faces': source_faces,
                'target_vision_frame': target_vision_frame,
                'target_frame_number': -1
            })
        write_image(output_path, result_frame)

    def get_model_options(self) -> ModelOptions:
        face_swapper_model = state_manager.get_item(self.model_key)
        face_swapper_model = 'inswapper_128' if has_execution_provider(
            'coreml') and face_swapper_model == 'inswapper_128_fp16' else face_swapper_model
        return self.MODEL_SET.get(face_swapper_model)
    
    def get_cached_model_options(self) -> ModelOptions:
        """Get model options with caching to avoid repeated lookups"""
        current_model = state_manager.get_item(self.model_key)
        if self._cached_model_options is None or self._cached_model_key != current_model:
            self._cached_model_options = self.get_model_options()
            self._cached_model_key = current_model
            self._cached_input_names = None  # Reset input names cache when model changes
        return self._cached_model_options
    
    def get_cached_input_names(self) -> dict:
        """Get face_swapper input names with caching"""
        if self._cached_input_names is None:
            face_swapper = self.get_inference_pool().get('face_swapper')
            self._cached_input_names = {inp.name: inp.name for inp in face_swapper.get_inputs()}
        return self._cached_input_names
    
    def clear_model_cache(self) -> None:
        """Clear cached model options"""
        self._cached_model_options = None
        self._cached_model_key = None
        self._cached_input_names = None

    def get_inference_pool(self) -> InferencePool:
        model_sources = self.get_model_options().get('sources')
        model_context = __name__ + '.' + state_manager.get_item(self.model_key)
        return inference_manager.get_inference_pool(model_context, model_sources)

    def clear_inference_pool(self) -> None:
        model_context = __name__ + '.' + state_manager.get_item(self.model_key)
        inference_manager.clear_inference_pool(model_context)

    def swap_face(self, source_face: Face, target_face: Face, temp_vision_frame: VisionFrame,
                  src_idx: int, target_frame_number: int, padding: Padding,
                  cached_settings: dict = None, masker: 'FaceMasker' = None) -> VisionFrame:
        """
        Swap face with optional cached settings to avoid repeated state lookups.
        cached_settings can contain: face_mask_types, face_mask_blur, face_mask_regions, face_mask_areas,
                                     auto_padding_model, pixel_boost_size
        """
        if masker is None:
            masker = FaceMasker()
        model_options = self.get_cached_model_options()
        model_template = model_options.get('template')
        model_size = model_options.get('size')
        
        # Use cached settings if provided, otherwise lookup from state
        if cached_settings:
            pixel_boost_size = cached_settings.get('pixel_boost_size')
            face_mask_types = cached_settings.get('face_mask_types')
            face_mask_blur = cached_settings.get('face_mask_blur') or 0.3
            face_mask_regions = cached_settings.get('face_mask_regions')
            face_mask_areas = cached_settings.get('face_mask_areas')
            auto_padding_model = cached_settings.get('auto_padding_model')
        else:
            pixel_boost_size = unpack_resolution(state_manager.get_item('face_swapper_pixel_boost'))
            face_mask_types = state_manager.get_item('face_mask_types')
            face_mask_blur = state_manager.get_item('face_mask_blur') or 0.3
            face_mask_regions = state_manager.get_item('face_mask_regions')
            face_mask_areas = state_manager.get_item('face_mask_areas')
            auto_padding_model = state_manager.get_item('auto_padding_model')

        pixel_boost_total = pixel_boost_size[0] // model_size[0]
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame,
                                                                        target_face.landmark_set.get('5/68'),
                                                                        model_template, pixel_boost_size)
        temp_vision_frames = []
        crop_masks = []

        if 'box' in face_mask_types:
            if auto_padding_model and auto_padding_model != "None":
                auto_data = getattr(target_face, 'auto_padding_data', None) or {}
                if auto_data.get('padding_needed'):
                    effective_padding = auto_data.get('recommended_padding') or padding
                else:
                    effective_padding = padding
            else:
                effective_padding = padding
            
            box_mask = masker.create_static_box_mask(crop_vision_frame.shape[:2][::-1],
                                                     face_mask_blur,
                                                     effective_padding)
            crop_masks.append(box_mask)

        if 'occlusion' in face_mask_types:
            occlusion_mask = masker.create_occlusion_mask(crop_vision_frame)
            crop_masks.append(occlusion_mask)

        pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
        
        # Batch prepare all tiles at once
        prepared_frames = self.batch_prepare_crop_frames(pixel_boost_vision_frames, model_options)
        
        # Run inference on each tile (cannot be batched due to model limitations)
        swapped_frames = []
        for prepared_frame in prepared_frames:
            swapped_frame = self.forward_swap_face_cached(source_face, target_face, prepared_frame, src_idx, model_options)
            swapped_frames.append(swapped_frame)
        
        # Batch normalize all results
        temp_vision_frames = self.batch_normalize_crop_frames(swapped_frames, model_options)
        
        crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)

        effective_face_mask_areas, apply_area_mask = masker.resolve_effective_face_mask_areas(
            target_face, face_mask_areas, face_mask_types)
        if apply_area_mask and effective_face_mask_areas:
            face_landmark_68 = cv2.transform(
                target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
            area_mask = masker.create_area_mask(crop_vision_frame, face_landmark_68, effective_face_mask_areas)
            crop_masks.append(area_mask)

        exclude_areas, apply_exclude = masker.resolve_auto_exclude_face_mask_areas(target_face)
        if apply_exclude and exclude_areas:
            face_landmark_68 = cv2.transform(
                target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
            exclude_mask = masker.create_area_mask(crop_vision_frame, face_landmark_68, exclude_areas)
            crop_masks.append(1.0 - exclude_mask)

        if 'region' in face_mask_types:
            region_mask = masker.create_region_mask(crop_vision_frame, face_mask_regions)
            crop_masks.append(region_mask)

        crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
        temp_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
        return temp_vision_frame

    def forward_swap_face(self, source_face: Face, target_face: Face, crop_vision_frame: VisionFrame, src_idx: int) -> VisionFrame:
        return self.forward_swap_face_cached(source_face, target_face, crop_vision_frame, src_idx, self.get_cached_model_options())

    def forward_swap_face_cached(self, source_face: Face, target_face: Face, crop_vision_frame: VisionFrame, src_idx: int, model_options: ModelOptions) -> VisionFrame:
        """Forward swap with cached model options to avoid repeated lookups"""
        face_swapper = self.get_inference_pool().get('face_swapper')
        model_type = model_options.get('type')
        input_names = self.get_cached_input_names()
        face_swapper_inputs = {}

        if 'source' in input_names:
            if model_type == 'blendswap' or model_type == 'uniface':
                face_swapper_inputs['source'] = self.prepare_source_frame(source_face, src_idx)
            else:
                source_embedding = self.prepare_source_embedding(source_face, src_idx)
                face_swapper_inputs['source'] = self.balance_source_embedding(source_embedding, target_face.embedding, model_type)
        if 'target' in input_names:
            face_swapper_inputs['target'] = crop_vision_frame

        with conditional_thread_semaphore():
            crop_vision_frame = face_swapper.run(None, face_swapper_inputs)[0][0]
        return crop_vision_frame

    def prepare_source_frame(self, source_face: Face, src_idx: int) -> VisionFrame:
        if src_idx in self.src_cache:
            return self.src_cache[src_idx]
        model_type = self.get_model_options().get('type')
        
        # Use correct source paths based on src_idx
        if src_idx == 1:
            source_paths = state_manager.get_item('source_paths_2')
        else:
            source_paths = state_manager.get_item('source_paths')
        
        if not source_paths:
            # Fallback to source_frame_dict if direct paths not available
            source_frame_dict = state_manager.get_item('source_frame_dict') or {}
            source_paths = source_frame_dict.get(src_idx, [])
        
        source_vision_frame = read_static_image(get_first(source_paths)) if source_paths else None
        if source_vision_frame is None:
            return None

        if model_type == 'blendswap':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame,
                                                                  source_face.landmark_set.get('5/68'),
                                                                  'arcface_112_v2', (112, 112))
        if model_type == 'uniface':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame,
                                                                  source_face.landmark_set.get('5/68'),
                                                                  'ffhq_512', (256, 256))
        source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
        source_vision_frame = source_vision_frame.transpose(2, 0, 1)
        source_vision_frame = numpy.expand_dims(source_vision_frame, axis=0).astype(numpy.float32)
        self.src_cache[src_idx] = source_vision_frame
        return source_vision_frame

    def prepare_source_embedding(self, source_face: Face, src_idx) -> Embedding:
        if src_idx in self.src_cache:
            return self.src_cache[src_idx]
        model_type = self.get_model_options().get('type')

        if model_type == 'ghost':
            source_embedding, _ = self.convert_embedding(source_face)
            source_embedding = source_embedding.reshape(1, -1)
        elif model_type == 'hyperswap':
            source_embedding = source_face.normed_embedding.reshape((1, -1))
        elif model_type == 'inswapper':
            model_path = self.get_model_options().get('sources').get('face_swapper').get('path')
            model_initializer = get_static_model_initializer(model_path)
            source_embedding = source_face.embedding.reshape((1, -1))
            source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
        else:
            _, source_normed_embedding = self.convert_embedding(source_face)
            source_embedding = source_normed_embedding.reshape(1, -1)
        self.src_cache[src_idx] = source_embedding
        return source_embedding

    def balance_source_embedding(self, source_embedding: Embedding, target_embedding: Embedding, model_type: str) -> Embedding:
        face_swapper_weight = state_manager.get_item('face_swapper_weight')
        if face_swapper_weight is None:
            face_swapper_weight = 0.5
        face_swapper_weight = numpy.interp(face_swapper_weight, [0, 1], [0.35, -0.35]).astype(numpy.float32)

        if model_type in ['hififace', 'hyperswap', 'inswapper', 'simswap']:
            target_embedding = target_embedding / numpy.linalg.norm(target_embedding)

        source_embedding = source_embedding.reshape(1, -1)
        target_embedding = target_embedding.reshape(1, -1)
        source_embedding = source_embedding * (1 - face_swapper_weight) + target_embedding * face_swapper_weight
        return source_embedding

    def convert_embedding(self, source_face: Face) -> Tuple[Embedding, Embedding]:
        embedding = source_face.embedding.reshape(-1, 512)
        embedding = self.forward_convert_embedding(embedding)
        embedding = embedding.ravel()
        normed_embedding = embedding / numpy.linalg.norm(embedding)
        return embedding, normed_embedding

    def forward_convert_embedding(self, embedding: Embedding) -> Embedding:
        embedding_converter = self.get_inference_pool().get('embedding_converter')

        with conditional_thread_semaphore():
            embedding = embedding_converter.run(None,
                                                {
                                                    'input': embedding
                                                })[0]

        return embedding

    def prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        return self.prepare_crop_frame_cached(crop_vision_frame, self.get_cached_model_options())
    
    def prepare_crop_frame_cached(self, crop_vision_frame: VisionFrame, model_options: ModelOptions) -> VisionFrame:
        """Prepare crop frame with cached model options"""
        model_mean = model_options.get('mean')
        model_standard_deviation = model_options.get('standard_deviation')

        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(numpy.float32)
        return crop_vision_frame
    
    def batch_prepare_crop_frames(self, crop_frames: List[VisionFrame], model_options: ModelOptions) -> List[VisionFrame]:
        """Batch prepare multiple crop frames with vectorized operations"""
        if len(crop_frames) == 0:
            return []
        
        model_mean = numpy.array(model_options.get('mean'), dtype=numpy.float32)
        model_std = numpy.array(model_options.get('standard_deviation'), dtype=numpy.float32)
        
        # Stack all frames: (N, H, W, C)
        stacked = numpy.stack(crop_frames, axis=0)
        
        # BGR -> RGB and normalize: vectorized
        stacked = stacked[:, :, :, ::-1] / 255.0
        stacked = (stacked - model_mean) / model_std
        
        # Transpose to (N, C, H, W) and convert to float32
        stacked = stacked.transpose(0, 3, 1, 2).astype(numpy.float32)
        
        # Split back into list with batch dim for ONNX: each is (1, C, H, W)
        return [stacked[i:i+1] for i in range(stacked.shape[0])]

    def normalize_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        return self.normalize_crop_frame_cached(crop_vision_frame, self.get_cached_model_options())
    
    def normalize_crop_frame_cached(self, crop_vision_frame: VisionFrame, model_options: ModelOptions) -> VisionFrame:
        """Normalize crop frame with cached model options"""
        model_type = model_options.get('type')
        model_mean = model_options.get('mean')
        model_standard_deviation = model_options.get('standard_deviation')

        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        if model_type in ['ghost', 'hififace', 'hyperswap', 'uniface']:
            crop_vision_frame = crop_vision_frame * model_standard_deviation + model_mean
        crop_vision_frame = crop_vision_frame.clip(0, 1)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
        return crop_vision_frame
    
    def batch_normalize_crop_frames(self, crop_frames: List[VisionFrame], model_options: ModelOptions) -> List[VisionFrame]:
        """Batch normalize multiple crop frames with vectorized operations"""
        if len(crop_frames) == 0:
            return []
        
        model_type = model_options.get('type')
        model_mean = numpy.array(model_options.get('mean'), dtype=numpy.float32)
        model_std = numpy.array(model_options.get('standard_deviation'), dtype=numpy.float32)
        
        # Stack all frames: each is (C, H, W), stack to (N, C, H, W)
        stacked = numpy.stack(crop_frames, axis=0)
        
        # Transpose to (N, H, W, C)
        stacked = stacked.transpose(0, 2, 3, 1)
        
        # Denormalize for ghost/uniface models
        if model_type in ('ghost', 'hififace', 'hyperswap', 'uniface'):
            stacked = stacked * model_std + model_mean
        
        # Clip and convert RGB -> BGR, scale to 255
        stacked = stacked.clip(0, 1)
        stacked = stacked[:, :, :, ::-1] * 255
        
        # Split back into list
        return [stacked[i] for i in range(stacked.shape[0])]
