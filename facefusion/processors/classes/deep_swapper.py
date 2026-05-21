from argparse import ArgumentParser
from typing import Any, List, Tuple

import cv2
import numpy
from cv2.typing import Size
from numpy.typing import NDArray

from facefusion import config, logger, process_manager, state_manager, wording
from facefusion.common_helper import create_int_metavar
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.jobs import job_store
from facefusion.processors import choices as processors_choices
from facefusion.processors.base_processor import BaseProcessor
from facefusion.processors.typing import DeepSwapperInputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import thread_semaphore
from facefusion.typing import ApplyStateItem, Args, Face, Mask, ModelSet, ProcessMode, QueuePayload, VisionFrame
from facefusion.vision import normalize_resolution, read_image, read_static_image, write_image
from facefusion.workers.classes.face_masker import FaceMasker


DRUUZIL_MODEL_NAMES = [
    'adam_levine_320',
    'adrianne_palicki_384',
    'agnetha_falskog_224',
    'alan_ritchson_320',
    'alicia_vikander_320',
    'amber_midthunder_320',
    'andras_arato_384',
    'andrew_tate_320',
    'angelina_jolie_384',
    'anne_hathaway_320',
    'anya_chalotra_320',
    'arnold_schwarzenegger_320',
    'benjamin_affleck_320',
    'benjamin_stiller_384',
    'bradley_pitt_224',
    'brie_larson_384',
    'bruce_campbell_384',
    'bryan_cranston_320',
    'catherine_blanchett_352',
    'christian_bale_320',
    'christopher_hemsworth_320',
    'christoph_waltz_384',
    'cillian_murphy_320',
    'cobie_smulders_256',
    'dwayne_johnson_384',
    'edward_norton_320',
    'elisabeth_shue_320',
    'elizabeth_olsen_384',
    'elon_musk_320',
    'emily_blunt_320',
    'emma_stone_384',
    'emma_watson_320',
    'erin_moriarty_384',
    'eva_green_320',
    'ewan_mcgregor_320',
    'florence_pugh_320',
    'freya_allan_320',
    'gary_cole_224',
    'gigi_hadid_224',
    'harrison_ford_384',
    'hayden_christensen_320',
    'heath_ledger_320',
    'henry_cavill_448',
    'hugh_jackman_384',
    'idris_elba_320',
    'jack_nicholson_320',
    'james_carrey_384',
    'james_mcavoy_320',
    'james_varney_320',
    'jason_momoa_320',
    'jason_statham_320',
    'jennifer_connelly_384',
    'jimmy_donaldson_320',
    'jordan_peterson_384',
    'karl_urban_224',
    'kate_beckinsale_384',
    'laurence_fishburne_384',
    'lili_reinhart_320',
    'luke_evans_384',
    'mads_mikkelsen_384',
    'mary_winstead_320',
    'margaret_qualley_384',
    'melina_juergens_320',
    'michael_fassbender_320',
    'michael_fox_320',
    'millie_bobby_brown_320',
    'morgan_freeman_320',
    'patrick_stewart_224',
    'rachel_weisz_384',
    'rebecca_ferguson_320',
    'scarlett_johansson_320',
    'shannen_doherty_384',
    'seth_macfarlane_384',
    'thomas_cruise_320',
    'thomas_hanks_384',
    'william_murray_384',
    'zoe_saldana_384',
]


def build_deep_swapper_model_entry(model_scope: str, model_name: str) -> dict:
    base_name = 'deepfacelive-models-' + model_scope
    return {
        'hashes': {
            'deep_swapper': {
                'url': 'https://huggingface.co/facefusion/' + base_name + '/resolve/main/' + model_name + '.hash',
                'path': resolve_relative_path('../.assets/models/' + model_scope + '/' + model_name + '.hash'),
            }
        },
        'sources': {
            'deep_swapper': {
                'url': 'https://huggingface.co/facefusion/' + base_name + '/resolve/main/' + model_name + '.dfm',
                'path': resolve_relative_path('../.assets/models/' + model_scope + '/' + model_name + '.dfm'),
            }
        },
        'template': 'dfl_whole_face',
    }


def build_model_set() -> ModelSet:
    model_set: ModelSet = {}
    for model_name in DRUUZIL_MODEL_NAMES:
        model_id = 'druuzil/' + model_name
        model_set[model_id] = build_deep_swapper_model_entry('druuzil', model_name)
    return model_set


def blend_frame(source_vision_frame: VisionFrame, target_vision_frame: VisionFrame, blend_factor: float) -> VisionFrame:
    return cv2.addWeighted(source_vision_frame, 1 - blend_factor, target_vision_frame, blend_factor, 0)


def equalize_frame_color(source_vision_frame: VisionFrame, target_vision_frame: VisionFrame, size: Size) -> VisionFrame:
    source_frame_resize = cv2.resize(source_vision_frame, size, interpolation=cv2.INTER_AREA).astype(numpy.float32)
    target_frame_resize = cv2.resize(target_vision_frame, size, interpolation=cv2.INTER_AREA).astype(numpy.float32)
    color_difference_vision_frame = numpy.subtract(source_frame_resize, target_frame_resize)
    color_difference_vision_frame = cv2.resize(
        color_difference_vision_frame, target_vision_frame.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    target_vision_frame = numpy.add(target_vision_frame, color_difference_vision_frame).clip(0, 255).astype(numpy.uint8)
    return target_vision_frame


def match_frame_color(source_vision_frame: VisionFrame, target_vision_frame: VisionFrame) -> VisionFrame:
    color_difference_sizes = numpy.linspace(16, target_vision_frame.shape[0], 3, endpoint=False)
    for color_difference_size in color_difference_sizes:
        source_vision_frame = equalize_frame_color(
            source_vision_frame, target_vision_frame, normalize_resolution((color_difference_size, color_difference_size)))
    target_vision_frame = equalize_frame_color(
        source_vision_frame, target_vision_frame, target_vision_frame.shape[:2][::-1])
    return target_vision_frame


def calculate_histogram_difference(source_vision_frame: VisionFrame, target_vision_frame: VisionFrame) -> float:
    histogram_source = cv2.calcHist(
        [cv2.cvtColor(source_vision_frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    histogram_target = cv2.calcHist(
        [cv2.cvtColor(target_vision_frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return float(numpy.interp(
        cv2.compareHist(histogram_source, histogram_target, cv2.HISTCMP_CORREL), [-1, 1], [0, 1]))


def conditional_match_frame_color(source_vision_frame: VisionFrame, target_vision_frame: VisionFrame) -> VisionFrame:
    histogram_factor = calculate_histogram_difference(source_vision_frame, target_vision_frame)
    return blend_frame(target_vision_frame, match_frame_color(source_vision_frame, target_vision_frame), histogram_factor)


def prepare_crop_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = cv2.addWeighted(crop_vision_frame, 1.75, cv2.GaussianBlur(crop_vision_frame, (0, 0), 2), -0.75, 0)
    crop_vision_frame = crop_vision_frame / 255.0
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(numpy.float32)
    return crop_vision_frame


def normalize_crop_frame(crop_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame = (crop_vision_frame * 255.0).clip(0, 255)
    return crop_vision_frame.astype(numpy.uint8)


def prepare_crop_mask(crop_source_mask: Mask, crop_target_mask: Mask, model_size: Size) -> Mask:
    blur_size = 6.25
    kernel_size = 3
    crop_mask = numpy.minimum.reduce([crop_source_mask, crop_target_mask])
    crop_mask = crop_mask.reshape(model_size).clip(0, 1)
    crop_mask = cv2.erode(crop_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=2)
    crop_mask = cv2.GaussianBlur(crop_mask, (0, 0), blur_size)
    return crop_mask


class DeepSwapper(BaseProcessor):
    MODEL_SET: ModelSet = build_model_set()
    model_key = 'deep_swapper_model'
    default_model = 'druuzil/elon_musk_320'
    priority = 1

    def register_args(self, program: ArgumentParser) -> None:
        group_processors = find_argument_group(program, 'processors')
        if group_processors:
            group_processors.add_argument(
                '--deep-swapper-model',
                help=wording.get('help.deep_swapper_model'),
                default=config.get_str_value('processors.deep_swapper_model', self.default_model),
                choices=self.list_models(),
            )
            group_processors.add_argument(
                '--deep-swapper-morph',
                help=wording.get('help.deep_swapper_morph'),
                type=int,
                default=config.get_int_value('processors.deep_swapper_morph', '100'),
                choices=processors_choices.deep_swapper_morph_range,
                metavar=create_int_metavar(processors_choices.deep_swapper_morph_range),
            )
            job_store.register_step_keys(['deep_swapper_model', 'deep_swapper_morph'])

    def apply_args(self, args: Args, apply_state_item: ApplyStateItem) -> None:
        apply_state_item('deep_swapper_model', args.get('deep_swapper_model'))
        apply_state_item('deep_swapper_morph', args.get('deep_swapper_morph'))

    def pre_process(self, mode: ProcessMode) -> bool:
        if mode in ['output', 'preview']:
            target_path = state_manager.get_item('target_path')
            if not is_image(target_path) and not is_video(target_path):
                logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
                return False
        if mode == 'output':
            output_path = state_manager.get_item('output_path')
            if not in_directory(output_path):
                logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
                return False
            if not same_file_extension([state_manager.get_item('target_path'), output_path]):
                logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
                return False
        return True

    def post_process(self) -> None:
        read_static_image.cache_clear()
        super().post_process()

    def get_model_size(self) -> Size:
        deep_swapper = self.get_inference_pool().get('deep_swapper')
        for deep_swapper_input in deep_swapper.get_inputs():
            if deep_swapper_input.name == 'in_face:0':
                return deep_swapper_input.shape[1:3]
        return 0, 0

    def process_frame(self, inputs: DeepSwapperInputs) -> VisionFrame:
        reference_faces = inputs.get('reference_faces') or {}
        target_vision_frame = inputs.get('target_vision_frame')
        face_selector_mode = state_manager.get_item('face_selector_mode')

        cached_faces = inputs.get('cached_faces')
        if cached_faces is not None:
            many_faces = cached_faces
        else:
            many_faces = sort_and_filter_faces(get_many_faces([target_vision_frame], is_target_frame=True))

        if face_selector_mode == 'one':
            target_face = get_one_face(many_faces)
            if target_face:
                target_vision_frame = self.swap_face(target_face, target_vision_frame)
        elif face_selector_mode == 'reference':
            reference_face_distance = state_manager.get_item('reference_face_distance')
            for ref_faces in reference_faces.values():
                if not ref_faces:
                    continue
                similar_faces = find_similar_faces(many_faces, ref_faces, reference_face_distance)
                for similar_face in similar_faces:
                    target_vision_frame = self.swap_face(similar_face, target_vision_frame)

        return target_vision_frame

    def process_frames(self, queue_payloads: List[QueuePayload]) -> List[Tuple[int, str]]:
        output_frames = []
        for queue_payload in process_manager.manage(queue_payloads):
            target_vision_path = queue_payload['frame_path']
            target_frame_number = queue_payload['frame_number']
            reference_faces = queue_payload['reference_faces']
            source_faces = queue_payload['source_faces']
            target_vision_frame = read_image(target_vision_path)

            process_inputs: DeepSwapperInputs = {
                'reference_faces': reference_faces,
                'source_faces': source_faces,
                'target_vision_frame': target_vision_frame,
                'target_frame_number': target_frame_number,
            }
            if 'cached_faces' in queue_payload:
                process_inputs['cached_faces'] = queue_payload['cached_faces']

            result_frame = self.process_frame(process_inputs)
            write_image(target_vision_path, result_frame)
            output_frames.append((target_frame_number, target_vision_path))
        return output_frames

    def process_image(self, target_path: str, output_path: str, reference_faces=None) -> None:
        if reference_faces is None:
            reference_faces = (
                get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else (None, None))
        target_vision_frame = read_static_image(target_path)
        output_vision_frame = self.process_frame({
            'reference_faces': reference_faces,
            'source_faces': {},
            'target_vision_frame': target_vision_frame,
            'target_frame_number': -1,
        })
        write_image(output_path, output_vision_frame)

    def swap_face(self, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
        masker = FaceMasker()
        model_template = self.get_model_options().get('template')
        model_size = self.get_model_size()
        face_mask_types = state_manager.get_item('face_mask_types') or []
        face_mask_blur = state_manager.get_item('face_mask_blur')
        face_mask_padding = state_manager.get_item('face_mask_padding')
        face_mask_regions = state_manager.get_item('face_mask_regions')
        face_mask_areas = state_manager.get_item('face_mask_areas')

        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
        crop_vision_frame_raw = crop_vision_frame.copy()
        crop_masks = []

        if 'box' in face_mask_types:
            box_mask = masker.create_static_box_mask(crop_vision_frame.shape[:2][::-1], face_mask_blur, face_mask_padding)
            crop_masks.append(box_mask)

        if 'occlusion' in face_mask_types:
            crop_masks.append(masker.create_occlusion_mask(crop_vision_frame))

        crop_vision_frame = prepare_crop_frame(crop_vision_frame)
        deep_swapper_morph = numpy.array([
            numpy.interp(state_manager.get_item('deep_swapper_morph'), [0, 100], [0, 1])
        ]).astype(numpy.float32)
        crop_vision_frame, crop_source_mask, crop_target_mask = self.forward(crop_vision_frame, deep_swapper_morph)
        crop_vision_frame = normalize_crop_frame(crop_vision_frame)
        crop_vision_frame = conditional_match_frame_color(crop_vision_frame_raw, crop_vision_frame)
        crop_masks.append(prepare_crop_mask(crop_source_mask, crop_target_mask, model_size))

        effective_face_mask_areas, apply_area_mask = masker.resolve_effective_face_mask_areas(
            target_face, face_mask_areas, face_mask_types)
        if apply_area_mask and effective_face_mask_areas:
            face_landmark_68 = cv2.transform(
                target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
            crop_masks.append(masker.create_area_mask(crop_vision_frame, face_landmark_68, effective_face_mask_areas))

        if 'region' in face_mask_types:
            crop_masks.append(masker.create_region_mask(crop_vision_frame, face_mask_regions))

        crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
        return paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)

    def forward(
        self, crop_vision_frame: VisionFrame, deep_swapper_morph: NDArray[Any]
    ) -> Tuple[VisionFrame, Mask, Mask]:
        deep_swapper = self.get_inference_pool().get('deep_swapper')
        deep_swapper_inputs = {}

        for deep_swapper_input in deep_swapper.get_inputs():
            if deep_swapper_input.name == 'in_face:0':
                deep_swapper_inputs[deep_swapper_input.name] = crop_vision_frame
            if deep_swapper_input.name == 'morph_value:0':
                deep_swapper_inputs[deep_swapper_input.name] = deep_swapper_morph

        with thread_semaphore():
            crop_target_mask, crop_vision_frame, crop_source_mask = deep_swapper.run(None, deep_swapper_inputs)

        return crop_vision_frame[0], crop_source_mask[0], crop_target_mask[0]
