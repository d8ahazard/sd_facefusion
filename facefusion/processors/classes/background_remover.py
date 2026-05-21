from argparse import ArgumentParser
from functools import partial
from typing import List, Tuple

import cv2
import numpy

from facefusion import config, logger, process_manager, state_manager, wording
from facefusion.common_helper import is_macos, is_windows
from facefusion.execution import has_execution_provider
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.jobs import job_store
from facefusion.normalizer import normalize_color
from facefusion.processors import choices as processors_choices
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import BackgroundRemoverInputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import thread_semaphore
from facefusion.typing import ApplyStateItem, Args, Mask, ModelSet, ProcessMode, QueuePayload, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image


def build_background_remover_model(
        model_name: str,
        release: str,
        model_type: str,
        size: Tuple[int, int],
        mean: List[float],
        standard_deviation: List[float],
) -> dict:
    base_url = f'https://github.com/facefusion/facefusion-assets/releases/download/{release}/{model_name}'
    return {
        'hashes': {
            'background_remover': {
                'url': f'{base_url}.hash',
                'path': resolve_relative_path(f'../.assets/models/{model_name}.hash'),
            }
        },
        'sources': {
            'background_remover': {
                'url': f'{base_url}.onnx',
                'path': resolve_relative_path(f'../.assets/models/{model_name}.onnx'),
            }
        },
        'type': model_type,
        'size': size,
        'mean': mean,
        'standard_deviation': standard_deviation,
    }


def sanitize_color_channel(value: int) -> int:
    return max(processors_choices.background_remover_color_range[0],
               min(processors_choices.background_remover_color_range[-1], int(value)))


class BackgroundRemover(BaseProcessor):
    MODEL_SET: ModelSet = {
        'ben_2': build_background_remover_model('ben_2', 'models-3.5.0', 'ben', (1024, 1024), [0.0, 0.0, 0.0],
                                                [1.0, 1.0, 1.0]),
        'birefnet_general': build_background_remover_model('birefnet_general', 'models-3.5.0', 'birefnet', (1024, 1024),
                                                           [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        'birefnet_portrait': build_background_remover_model('birefnet_portrait', 'models-3.5.0', 'birefnet', (1024, 1024),
                                                            [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        'corridor_key_1024': build_background_remover_model('corridor_key_1024', 'models-3.6.0', 'corridor_key',
                                                            (1024, 1024), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'corridor_key_2048': build_background_remover_model('corridor_key_2048', 'models-3.6.0', 'corridor_key',
                                                            (2048, 2048), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'isnet_general': build_background_remover_model('isnet_general', 'models-3.5.0', 'isnet', (1024, 1024),
                                                        [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        'modnet': build_background_remover_model('modnet', 'models-3.5.0', 'modnet', (512, 512), [0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5]),
        'ormbg': build_background_remover_model('ormbg', 'models-3.5.0', 'ormbg', (1024, 1024), [0.0, 0.0, 0.0],
                                                [1.0, 1.0, 1.0]),
        'rmbg_1.4': build_background_remover_model('rmbg_1.4', 'models-3.5.0', 'rmbg', (1024, 1024), [0.5, 0.5, 0.5],
                                                   [1.0, 1.0, 1.0]),
        'rmbg_2.0': build_background_remover_model('rmbg_2.0', 'models-3.5.0', 'rmbg', (1024, 1024),
                                                   [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'silueta': build_background_remover_model('silueta', 'models-3.5.0', 'silueta', (320, 320),
                                                  [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'u2net_cloth': build_background_remover_model('u2net_cloth', 'models-3.5.0', 'u2net_cloth', (768, 768),
                                                      [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'u2net_general': build_background_remover_model('u2net_general', 'models-3.5.0', 'u2net', (320, 320),
                                                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'u2net_human': build_background_remover_model('u2net_human', 'models-3.5.0', 'u2net', (320, 320),
                                                      [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'u2netp': build_background_remover_model('u2netp', 'models-3.5.0', 'u2netp', (320, 320),
                                                 [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    }

    model_key = 'background_remover_model'
    default_model = 'modnet'
    is_face_processor = False

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, 'processors')
        if group_processors:
            group_processors.add_argument(
                '--background-remover-model',
                help=wording.get('help.background_remover_model'),
                default=config.get_str_value('processors.background_remover_model', self.default_model),
                choices=self.list_models(),
            )
            group_processors.add_argument(
                '--background-remover-fill-color',
                help=wording.get('help.background_remover_fill_color'),
                type=partial(sanitize_color_channel),
                default=config.get_int_list('processors.background_remover_fill_color', '0 0 0 0'),
                nargs='+',
            )
            group_processors.add_argument(
                '--background-remover-despill-color',
                help=wording.get('help.background_remover_despill_color'),
                type=partial(sanitize_color_channel),
                default=config.get_int_list('processors.background_remover_despill_color', '0 0 0 0'),
                nargs='+',
            )
            job_store.register_step_keys([
                'background_remover_model',
                'background_remover_fill_color',
                'background_remover_despill_color',
            ])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item('background_remover_model', args.get('background_remover_model'))
        apply_state_item('background_remover_fill_color', normalize_color(args.get('background_remover_fill_color')))
        apply_state_item('background_remover_despill_color', normalize_color(args.get('background_remover_despill_color')))

    def pre_process(self, mode: ProcessMode) -> bool:
        if mode in ['output', 'preview'] and not is_image(state_manager.get_item('target_path')) and not is_video(
                state_manager.get_item('target_path')):
            logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
            logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
            return False
        if mode == 'output' and not same_file_extension(
                [state_manager.get_item('target_path'), state_manager.get_item('output_path')]):
            logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
            return False
        return True

    def get_inference_pool(self):
        model_type = self.get_model_options().get('type')
        if is_macos() and has_execution_provider('coreml') or is_windows() and has_execution_provider(
                'directml') and model_type == 'corridor_key':
            previous_providers = state_manager.get_item('execution_providers')
            state_manager.set_item('execution_providers', ['cpu'])
            inference_pool = super().get_inference_pool()
            state_manager.set_item('execution_providers', previous_providers)
            return inference_pool
        return super().get_inference_pool()

    def remove_background(self, temp_vision_frame: VisionFrame) -> VisionFrame:
        model_type = self.get_model_options().get('type')

        if model_type == 'corridor_key':
            remove_vision_mask, remove_vision_frame = self.forward_corridor_key(self.prepare_temp_frame(temp_vision_frame))
            remove_vision_frame = numpy.squeeze(remove_vision_frame).transpose(1, 2, 0)
            remove_vision_frame = numpy.clip(remove_vision_frame * 255, 0, 255).astype(numpy.uint8)
            temp_vision_frame = cv2.resize(remove_vision_frame[:, :, ::-1], temp_vision_frame.shape[:2][::-1])
        else:
            remove_vision_mask = self.forward(self.prepare_temp_frame(temp_vision_frame))

        remove_vision_mask = self.normalize_vision_mask(remove_vision_mask)
        remove_vision_mask = cv2.resize(remove_vision_mask, temp_vision_frame.shape[:2][::-1])
        temp_vision_frame = self.apply_despill_color(temp_vision_frame)
        temp_vision_frame = self.apply_fill_color(temp_vision_frame, remove_vision_mask)
        return temp_vision_frame

    def forward(self, temp_vision_frame: VisionFrame) -> VisionFrame:
        background_remover = self.get_inference_pool().get('background_remover')
        model_type = self.get_model_options().get('type')

        with thread_semaphore():
            remove_vision_frame = background_remover.run(None, {
                'input': temp_vision_frame
            })[0]

            if model_type == 'u2net_cloth':
                remove_vision_frame = numpy.argmax(remove_vision_frame, axis=1)

        return remove_vision_frame

    def forward_corridor_key(self, temp_vision_frame: VisionFrame) -> Tuple[Mask, VisionFrame]:
        background_remover = self.get_inference_pool().get('background_remover')

        with thread_semaphore():
            remove_vision_mask, remove_vision_frame = background_remover.run(None, {
                'input': temp_vision_frame
            })

        return remove_vision_mask, remove_vision_frame

    def prepare_temp_frame(self, temp_vision_frame: VisionFrame) -> VisionFrame:
        model_type = self.get_model_options().get('type')
        model_size = self.get_model_options().get('size')
        model_mean = self.get_model_options().get('mean')
        model_standard_deviation = self.get_model_options().get('standard_deviation')

        if model_type == 'corridor_key':
            coarse_color = temp_vision_frame[:, :, ::-1].astype(numpy.float32) / 255.0
            coarse_bias = coarse_color[:, :, 1] - numpy.maximum(coarse_color[:, :, 0], coarse_color[:, :, 2])
            coarse_vision_mask = cv2.resize(1.0 - numpy.clip(coarse_bias * 2.0, 0, 1), model_size)[:, :, numpy.newaxis]

        temp_vision_frame = cv2.resize(temp_vision_frame, model_size)
        temp_vision_frame = temp_vision_frame[:, :, ::-1] / 255.0
        temp_vision_frame = (temp_vision_frame - model_mean) / model_standard_deviation

        if model_type == 'corridor_key':
            temp_vision_frame = numpy.concatenate([temp_vision_frame, coarse_vision_mask], axis=2)

        temp_vision_frame = temp_vision_frame.transpose(2, 0, 1)
        temp_vision_frame = numpy.expand_dims(temp_vision_frame, axis=0).astype(numpy.float32)
        return temp_vision_frame

    def normalize_vision_mask(self, temp_vision_mask: Mask) -> Mask:
        temp_vision_mask = numpy.squeeze(temp_vision_mask).clip(0, 1) * 255
        temp_vision_mask = numpy.clip(temp_vision_mask, 0, 255).astype(numpy.uint8)
        return temp_vision_mask

    def apply_fill_color(self, temp_vision_frame: VisionFrame, temp_vision_mask: Mask) -> VisionFrame:
        background_remover_fill_color = state_manager.get_item('background_remover_fill_color')
        temp_vision_mask = temp_vision_mask.astype(numpy.float32) / 255
        temp_vision_mask = numpy.expand_dims(temp_vision_mask, axis=2)
        temp_vision_mask = (1 - temp_vision_mask) * background_remover_fill_color[-1] / 255
        fill_vision_frame = numpy.zeros_like(temp_vision_frame)
        fill_vision_frame[:, :, 0] = background_remover_fill_color[2]
        fill_vision_frame[:, :, 1] = background_remover_fill_color[1]
        fill_vision_frame[:, :, 2] = background_remover_fill_color[0]
        temp_vision_frame = temp_vision_frame * (1 - temp_vision_mask) + fill_vision_frame * temp_vision_mask
        return temp_vision_frame.astype(numpy.uint8)

    def apply_despill_color(self, temp_vision_frame: VisionFrame) -> VisionFrame:
        background_remover_despill_color = state_manager.get_item('background_remover_despill_color')
        temp_vision_frame = temp_vision_frame.astype(numpy.float32)
        color_alpha = background_remover_despill_color[3] / 255.0
        despill_vision_frame = numpy.zeros_like(temp_vision_frame)
        despill_vision_frame[:, :, 0] = background_remover_despill_color[2]
        despill_vision_frame[:, :, 1] = background_remover_despill_color[1]
        despill_vision_frame[:, :, 2] = background_remover_despill_color[0]
        color_weight = despill_vision_frame / numpy.maximum(numpy.max(background_remover_despill_color[:3]), 1)
        color_limit = numpy.roll(temp_vision_frame, 1, 2) + numpy.roll(temp_vision_frame, -1, 2)
        limit_vision_frame = numpy.minimum(temp_vision_frame, color_limit * 0.5)
        temp_vision_frame = temp_vision_frame + (limit_vision_frame - temp_vision_frame) * color_alpha * color_weight
        return temp_vision_frame.astype(numpy.uint8)

    def process_frame(self, inputs: BackgroundRemoverInputs) -> VisionFrame:
        target_vision_frame = inputs.get('target_vision_frame')
        return self.remove_background(target_vision_frame)

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            target_vision_frame = read_image(target_vision_path)
            output_vision_frame = self.process_frame({
                'target_vision_frame': target_vision_frame,
            })
            write_image(target_vision_path, output_vision_frame)
            output_frames.append((queue_payload['frame_number'], target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str, _=None) -> None:
        target_vision_frame = read_static_image(target_path)
        output_vision_frame = self.process_frame({
            'target_vision_frame': target_vision_frame,
        })
        write_image(output_path, output_vision_frame)
