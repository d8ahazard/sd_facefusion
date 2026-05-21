from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import calc_int_step
from facefusion.processors import choices as processors_choices
from facefusion.processors.classes.background_remover import BackgroundRemover, sanitize_color_channel
from facefusion.processors.core import load_processor_module
from facefusion.processors.typing import BackgroundRemoverModel
from facefusion.uis.core import get_ui_component, register_ui_component

BACKGROUND_REMOVER_MODEL_DROPDOWN: Optional[gradio.Dropdown] = None
BACKGROUND_REMOVER_FILL_COLOR_WRAPPER: Optional[gradio.Group] = None
BACKGROUND_REMOVER_FILL_COLOR_RED_NUMBER: Optional[gradio.Number] = None
BACKGROUND_REMOVER_FILL_COLOR_GREEN_NUMBER: Optional[gradio.Number] = None
BACKGROUND_REMOVER_FILL_COLOR_BLUE_NUMBER: Optional[gradio.Number] = None
BACKGROUND_REMOVER_FILL_COLOR_ALPHA_NUMBER: Optional[gradio.Number] = None
BACKGROUND_REMOVER_DESPILL_COLOR_WRAPPER: Optional[gradio.Group] = None
BACKGROUND_REMOVER_DESPILL_COLOR_RED_NUMBER: Optional[gradio.Number] = None
BACKGROUND_REMOVER_DESPILL_COLOR_GREEN_NUMBER: Optional[gradio.Number] = None
BACKGROUND_REMOVER_DESPILL_COLOR_BLUE_NUMBER: Optional[gradio.Number] = None
BACKGROUND_REMOVER_DESPILL_COLOR_ALPHA_NUMBER: Optional[gradio.Number] = None
PROCESSOR_KEY = 'Background Remover'


def render() -> None:
    global BACKGROUND_REMOVER_MODEL_DROPDOWN
    global BACKGROUND_REMOVER_FILL_COLOR_WRAPPER
    global BACKGROUND_REMOVER_FILL_COLOR_RED_NUMBER
    global BACKGROUND_REMOVER_FILL_COLOR_GREEN_NUMBER
    global BACKGROUND_REMOVER_FILL_COLOR_BLUE_NUMBER
    global BACKGROUND_REMOVER_FILL_COLOR_ALPHA_NUMBER
    global BACKGROUND_REMOVER_DESPILL_COLOR_WRAPPER
    global BACKGROUND_REMOVER_DESPILL_COLOR_RED_NUMBER
    global BACKGROUND_REMOVER_DESPILL_COLOR_GREEN_NUMBER
    global BACKGROUND_REMOVER_DESPILL_COLOR_BLUE_NUMBER
    global BACKGROUND_REMOVER_DESPILL_COLOR_ALPHA_NUMBER

    has_background_remover = PROCESSOR_KEY in state_manager.get_item('processors')
    background_remover_fill_color = state_manager.get_item('background_remover_fill_color') or (0, 0, 0, 0)
    background_remover_despill_color = state_manager.get_item('background_remover_despill_color') or (0, 0, 0, 0)

    BACKGROUND_REMOVER_MODEL_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.background_remover_model_dropdown'),
        choices=BackgroundRemover().list_models(),
        value=state_manager.get_item('background_remover_model'),
        visible=has_background_remover,
    )
    with gradio.Group(visible=has_background_remover) as BACKGROUND_REMOVER_FILL_COLOR_WRAPPER:
        with gradio.Row():
            BACKGROUND_REMOVER_FILL_COLOR_RED_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_fill_color_red_number'),
                value=background_remover_fill_color[0],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )
            BACKGROUND_REMOVER_FILL_COLOR_GREEN_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_fill_color_green_number'),
                value=background_remover_fill_color[1],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )
        with gradio.Row():
            BACKGROUND_REMOVER_FILL_COLOR_BLUE_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_fill_color_blue_number'),
                value=background_remover_fill_color[2],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )
            BACKGROUND_REMOVER_FILL_COLOR_ALPHA_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_fill_color_alpha_number'),
                value=background_remover_fill_color[3],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )
    with gradio.Group(visible=has_background_remover) as BACKGROUND_REMOVER_DESPILL_COLOR_WRAPPER:
        with gradio.Row():
            BACKGROUND_REMOVER_DESPILL_COLOR_RED_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_despill_color_red_number'),
                value=background_remover_despill_color[0],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )
            BACKGROUND_REMOVER_DESPILL_COLOR_GREEN_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_despill_color_green_number'),
                value=background_remover_despill_color[1],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )
        with gradio.Row():
            BACKGROUND_REMOVER_DESPILL_COLOR_BLUE_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_despill_color_blue_number'),
                value=background_remover_despill_color[2],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )
            BACKGROUND_REMOVER_DESPILL_COLOR_ALPHA_NUMBER = gradio.Number(
                label=wording.get('uis.background_remover_despill_color_alpha_number'),
                value=background_remover_despill_color[3],
                minimum=processors_choices.background_remover_color_range[0],
                maximum=processors_choices.background_remover_color_range[-1],
                step=calc_int_step(processors_choices.background_remover_color_range),
            )

    register_ui_component('background_remover_model_dropdown', BACKGROUND_REMOVER_MODEL_DROPDOWN)
    register_ui_component('background_remover_fill_color_red_number', BACKGROUND_REMOVER_FILL_COLOR_RED_NUMBER)
    register_ui_component('background_remover_fill_color_green_number', BACKGROUND_REMOVER_FILL_COLOR_GREEN_NUMBER)
    register_ui_component('background_remover_fill_color_blue_number', BACKGROUND_REMOVER_FILL_COLOR_BLUE_NUMBER)
    register_ui_component('background_remover_fill_color_alpha_number', BACKGROUND_REMOVER_FILL_COLOR_ALPHA_NUMBER)
    register_ui_component('background_remover_despill_color_red_number', BACKGROUND_REMOVER_DESPILL_COLOR_RED_NUMBER)
    register_ui_component('background_remover_despill_color_green_number', BACKGROUND_REMOVER_DESPILL_COLOR_GREEN_NUMBER)
    register_ui_component('background_remover_despill_color_blue_number', BACKGROUND_REMOVER_DESPILL_COLOR_BLUE_NUMBER)
    register_ui_component('background_remover_despill_color_alpha_number', BACKGROUND_REMOVER_DESPILL_COLOR_ALPHA_NUMBER)


def listen() -> None:
    BACKGROUND_REMOVER_MODEL_DROPDOWN.change(
        update_background_remover_model,
        inputs=BACKGROUND_REMOVER_MODEL_DROPDOWN,
        outputs=BACKGROUND_REMOVER_MODEL_DROPDOWN,
    )
    background_remover_fill_color_inputs = [
        BACKGROUND_REMOVER_FILL_COLOR_RED_NUMBER,
        BACKGROUND_REMOVER_FILL_COLOR_GREEN_NUMBER,
        BACKGROUND_REMOVER_FILL_COLOR_BLUE_NUMBER,
        BACKGROUND_REMOVER_FILL_COLOR_ALPHA_NUMBER,
    ]
    background_remover_despill_color_inputs = [
        BACKGROUND_REMOVER_DESPILL_COLOR_RED_NUMBER,
        BACKGROUND_REMOVER_DESPILL_COLOR_GREEN_NUMBER,
        BACKGROUND_REMOVER_DESPILL_COLOR_BLUE_NUMBER,
        BACKGROUND_REMOVER_DESPILL_COLOR_ALPHA_NUMBER,
    ]

    for background_remover_fill_color_input in background_remover_fill_color_inputs:
        background_remover_fill_color_input.change(
            update_background_remover_fill_color,
            inputs=background_remover_fill_color_inputs,
        )

    for background_remover_despill_color_input in background_remover_despill_color_inputs:
        background_remover_despill_color_input.change(
            update_background_remover_despill_color,
            inputs=background_remover_despill_color_inputs,
        )

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(
            remote_update,
            inputs=processors_checkbox_group,
            outputs=[
                BACKGROUND_REMOVER_MODEL_DROPDOWN,
                BACKGROUND_REMOVER_FILL_COLOR_WRAPPER,
                BACKGROUND_REMOVER_DESPILL_COLOR_WRAPPER,
            ],
        )


def remote_update(processors: List[str]) -> Tuple[gradio.update, gradio.update, gradio.update]:
    has_background_remover = PROCESSOR_KEY in processors
    return (
        gradio.update(visible=has_background_remover),
        gradio.update(visible=has_background_remover),
        gradio.update(visible=has_background_remover),
    )


def update_background_remover_model(background_remover_model: BackgroundRemoverModel) -> gradio.update:
    background_remover_module = load_processor_module(PROCESSOR_KEY)
    background_remover_module.clear_inference_pool()
    state_manager.set_item('background_remover_model', background_remover_model)

    if background_remover_module.pre_check():
        return gradio.update(value=state_manager.get_item('background_remover_model'))
    return gradio.update()


def update_background_remover_fill_color(red: int, green: int, blue: int, alpha: int) -> None:
    state_manager.set_item('background_remover_fill_color', (
        sanitize_color_channel(red),
        sanitize_color_channel(green),
        sanitize_color_channel(blue),
        sanitize_color_channel(alpha),
    ))


def update_background_remover_despill_color(red: int, green: int, blue: int, alpha: int) -> None:
    state_manager.set_item('background_remover_despill_color', (
        sanitize_color_channel(red),
        sanitize_color_channel(green),
        sanitize_color_channel(blue),
        sanitize_color_channel(alpha),
    ))
