from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import calc_float_step, get_first
from facefusion.processors import choices as processors_choices
from facefusion.processors.classes.face_swapper import FaceSwapper
from facefusion.processors.core import load_processor_module
from facefusion.processors.typing import FaceSwapperModel
from facefusion.uis.core import get_ui_component, register_ui_component

FACE_SWAPPER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SWAPPER_PIXEL_BOOST_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SWAPPER_WEIGHT_SLIDER: Optional[gradio.Slider] = None
PROCESSOR_KEY = 'Face Swapper'


def has_face_swapper_weight() -> bool:
    return state_manager.get_item('face_swapper_model') in [
        'ghost_1_256', 'ghost_2_256', 'ghost_3_256', 'hififace_unofficial_256',
        'hyperswap_1a_256', 'hyperswap_1b_256', 'hyperswap_1c_256',
        'inswapper_128', 'inswapper_128_fp16', 'simswap_256', 'simswap_unofficial_512',
    ]


def render() -> None:
    global FACE_SWAPPER_MODEL_DROPDOWN
    global FACE_SWAPPER_PIXEL_BOOST_DROPDOWN
    global FACE_SWAPPER_WEIGHT_SLIDER

    has_face_swapper = PROCESSOR_KEY in state_manager.get_item('processors')
    FACE_SWAPPER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_swapper_model_dropdown'),
        choices=FaceSwapper().list_models(),
        value=state_manager.get_item('face_swapper_model'),
        visible=has_face_swapper,
    )
    FACE_SWAPPER_PIXEL_BOOST_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_swapper_pixel_boost_dropdown'),
        choices=processors_choices.face_swapper_set.get(state_manager.get_item('face_swapper_model')),
        value=state_manager.get_item('face_swapper_pixel_boost'),
        visible=has_face_swapper,
    )
    weight = state_manager.get_item('face_swapper_weight')
    if weight is None:
        weight = 0.5
    FACE_SWAPPER_WEIGHT_SLIDER = gradio.Slider(
        label=wording.get('uis.face_swapper_weight_slider'),
        value=weight,
        minimum=processors_choices.face_swapper_weight_range[0],
        maximum=processors_choices.face_swapper_weight_range[-1],
        step=calc_float_step(processors_choices.face_swapper_weight_range),
        visible=has_face_swapper and has_face_swapper_weight(),
    )
    register_ui_component('face_swapper_model_dropdown', FACE_SWAPPER_MODEL_DROPDOWN)
    register_ui_component('face_swapper_pixel_boost_dropdown', FACE_SWAPPER_PIXEL_BOOST_DROPDOWN)
    register_ui_component('face_swapper_weight_slider', FACE_SWAPPER_WEIGHT_SLIDER)


def listen() -> None:
    FACE_SWAPPER_MODEL_DROPDOWN.change(
        update_face_swapper_model,
        inputs=FACE_SWAPPER_MODEL_DROPDOWN,
        outputs=[FACE_SWAPPER_MODEL_DROPDOWN, FACE_SWAPPER_PIXEL_BOOST_DROPDOWN, FACE_SWAPPER_WEIGHT_SLIDER],
    )
    FACE_SWAPPER_PIXEL_BOOST_DROPDOWN.change(update_face_swapper_pixel_boost, inputs=FACE_SWAPPER_PIXEL_BOOST_DROPDOWN)
    FACE_SWAPPER_WEIGHT_SLIDER.change(update_face_swapper_weight, inputs=FACE_SWAPPER_WEIGHT_SLIDER)

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(
            remote_update,
            inputs=processors_checkbox_group,
            outputs=[FACE_SWAPPER_MODEL_DROPDOWN, FACE_SWAPPER_PIXEL_BOOST_DROPDOWN, FACE_SWAPPER_WEIGHT_SLIDER],
        )


def remote_update(processors: List[str]) -> Tuple[gradio.update, gradio.update, gradio.update]:
    has_face_swapper = PROCESSOR_KEY in processors
    return (
        gradio.update(visible=has_face_swapper),
        gradio.update(visible=has_face_swapper),
        gradio.update(visible=has_face_swapper and has_face_swapper_weight()),
    )


def update_face_swapper_model(face_swapper_model: FaceSwapperModel) -> Tuple[gradio.update, gradio.update, gradio.update]:
    face_swapper_module = load_processor_module(PROCESSOR_KEY)
    face_swapper_module.clear_inference_pool()
    face_swapper_module.clear_model_cache()
    state_manager.set_item('face_swapper_model', face_swapper_model)

    if face_swapper_module.pre_check():
        face_swapper_pixel_boost_choices = processors_choices.face_swapper_set.get(
            state_manager.get_item('face_swapper_model'))
        state_manager.set_item('face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices))
        return (
            gradio.update(value=state_manager.get_item('face_swapper_model')),
            gradio.update(
                value=state_manager.get_item('face_swapper_pixel_boost'),
                choices=face_swapper_pixel_boost_choices,
            ),
            gradio.update(visible=has_face_swapper_weight()),
        )
    return gradio.update(), gradio.update(), gradio.update()


def update_face_swapper_pixel_boost(face_swapper_pixel_boost: str) -> None:
    state_manager.set_item('face_swapper_pixel_boost', face_swapper_pixel_boost)


def update_face_swapper_weight(face_swapper_weight: float) -> None:
    state_manager.set_item('face_swapper_weight', face_swapper_weight)
