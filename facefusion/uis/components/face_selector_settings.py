"""Face selection filters and detection-related settings (Settings tab)."""

from typing import Optional

import gradio

import facefusion.choices
from facefusion import state_manager, wording
from facefusion.common_helper import calc_float_step, calc_int_step
from facefusion.uis.core import register_ui_component


FACE_SELECTOR_MODE_SETTINGS_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_ORDER_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_GENDER_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_RACE_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_AGE_RANGE_START_SLIDER: Optional[gradio.Slider] = None
FACE_SELECTOR_AGE_RANGE_END_SLIDER: Optional[gradio.Slider] = None
REFERENCE_FACE_DISTANCE_SLIDER: Optional[gradio.Slider] = None


def render() -> None:
    global FACE_SELECTOR_MODE_SETTINGS_DROPDOWN
    global FACE_SELECTOR_ORDER_DROPDOWN
    global FACE_SELECTOR_GENDER_DROPDOWN
    global FACE_SELECTOR_RACE_DROPDOWN
    global FACE_SELECTOR_AGE_RANGE_START_SLIDER
    global FACE_SELECTOR_AGE_RANGE_END_SLIDER
    global REFERENCE_FACE_DISTANCE_SLIDER

    gradio.Markdown('### Face selection')
    FACE_SELECTOR_MODE_SETTINGS_DROPDOWN = gradio.Dropdown(
        label=wording.get('uis.face_selector_mode_dropdown'),
        choices=facefusion.choices.face_selector_modes,
        value=state_manager.get_item('face_selector_mode'),
    )
    with gradio.Row():
        FACE_SELECTOR_ORDER_DROPDOWN = gradio.Dropdown(
            label=wording.get('uis.face_selector_order_dropdown'),
            choices=facefusion.choices.face_selector_orders,
            value=state_manager.get_item('face_selector_order'),
        )
        FACE_SELECTOR_GENDER_DROPDOWN = gradio.Dropdown(
            label=wording.get('uis.face_selector_gender_dropdown'),
            choices=['none'] + facefusion.choices.face_selector_genders,
            value=state_manager.get_item('face_selector_gender') or 'none',
        )
        FACE_SELECTOR_RACE_DROPDOWN = gradio.Dropdown(
            label=wording.get('uis.face_selector_race_dropdown'),
            choices=['none'] + facefusion.choices.face_selector_races,
            value=state_manager.get_item('face_selector_race') or 'none',
        )
    face_selector_age_start = state_manager.get_item('face_selector_age_start') or facefusion.choices.face_selector_age_range[0]
    face_selector_age_end = state_manager.get_item('face_selector_age_end') or facefusion.choices.face_selector_age_range[-1]
    with gradio.Row():
        FACE_SELECTOR_AGE_RANGE_START_SLIDER = gradio.Slider(
            label=wording.get('uis.face_selector_age_start_slider'),
            value=face_selector_age_start,
            step=calc_int_step(facefusion.choices.face_selector_age_range),
            minimum=facefusion.choices.face_selector_age_range[0],
            maximum=facefusion.choices.face_selector_age_range[-1],
        )
        FACE_SELECTOR_AGE_RANGE_END_SLIDER = gradio.Slider(
            label=wording.get('uis.face_selector_age_end_slider'),
            value=face_selector_age_end,
            step=calc_int_step(facefusion.choices.face_selector_age_range),
            minimum=facefusion.choices.face_selector_age_range[0],
            maximum=facefusion.choices.face_selector_age_range[-1],
        )
    REFERENCE_FACE_DISTANCE_SLIDER = gradio.Slider(
        label=wording.get('uis.reference_face_distance_slider'),
        value=state_manager.get_item('reference_face_distance'),
        step=calc_float_step(facefusion.choices.reference_face_distance_range),
        minimum=facefusion.choices.reference_face_distance_range[0],
        maximum=facefusion.choices.reference_face_distance_range[-1],
    )

    register_ui_component('face_selector_mode_settings_dropdown', FACE_SELECTOR_MODE_SETTINGS_DROPDOWN)
    register_ui_component('face_selector_order_dropdown', FACE_SELECTOR_ORDER_DROPDOWN)
    register_ui_component('face_selector_gender_dropdown', FACE_SELECTOR_GENDER_DROPDOWN)
    register_ui_component('face_selector_race_dropdown', FACE_SELECTOR_RACE_DROPDOWN)
    register_ui_component('face_selector_age_range_start_slider', FACE_SELECTOR_AGE_RANGE_START_SLIDER)
    register_ui_component('face_selector_age_range_end_slider', FACE_SELECTOR_AGE_RANGE_END_SLIDER)
    register_ui_component('reference_face_distance_slider', REFERENCE_FACE_DISTANCE_SLIDER)


def listen() -> None:
    from facefusion.uis.components import face_selector

    mapping_refresh_outputs = face_selector.get_mapping_refresh_output_components()
    mode_outputs = face_selector.get_mode_change_output_components()

    if FACE_SELECTOR_MODE_SETTINGS_DROPDOWN and mode_outputs:
        FACE_SELECTOR_MODE_SETTINGS_DROPDOWN.change(
            face_selector.apply_face_selector_mode,
            inputs=FACE_SELECTOR_MODE_SETTINGS_DROPDOWN,
            outputs=mode_outputs,
        )
    elif FACE_SELECTOR_MODE_SETTINGS_DROPDOWN:
        FACE_SELECTOR_MODE_SETTINGS_DROPDOWN.change(
            lambda m: state_manager.set_item('face_selector_mode', m),
            inputs=FACE_SELECTOR_MODE_SETTINGS_DROPDOWN,
        )

    if FACE_SELECTOR_ORDER_DROPDOWN and mapping_refresh_outputs:
        FACE_SELECTOR_ORDER_DROPDOWN.change(
            face_selector.update_face_selector_order,
            inputs=FACE_SELECTOR_ORDER_DROPDOWN,
            outputs=mapping_refresh_outputs,
        )
    if FACE_SELECTOR_GENDER_DROPDOWN and mapping_refresh_outputs:
        FACE_SELECTOR_GENDER_DROPDOWN.change(
            face_selector.update_face_selector_gender,
            inputs=FACE_SELECTOR_GENDER_DROPDOWN,
            outputs=mapping_refresh_outputs,
        )
    if FACE_SELECTOR_RACE_DROPDOWN and mapping_refresh_outputs:
        FACE_SELECTOR_RACE_DROPDOWN.change(
            face_selector.update_face_selector_race,
            inputs=FACE_SELECTOR_RACE_DROPDOWN,
            outputs=mapping_refresh_outputs,
        )

    if FACE_SELECTOR_AGE_RANGE_START_SLIDER and FACE_SELECTOR_AGE_RANGE_END_SLIDER and mapping_refresh_outputs:
        FACE_SELECTOR_AGE_RANGE_START_SLIDER.release(
            face_selector.update_face_selector_age_range,
            inputs=[FACE_SELECTOR_AGE_RANGE_START_SLIDER, FACE_SELECTOR_AGE_RANGE_END_SLIDER],
            outputs=mapping_refresh_outputs,
        )
        FACE_SELECTOR_AGE_RANGE_END_SLIDER.release(
            face_selector.update_face_selector_age_range,
            inputs=[FACE_SELECTOR_AGE_RANGE_START_SLIDER, FACE_SELECTOR_AGE_RANGE_END_SLIDER],
            outputs=mapping_refresh_outputs,
        )

    if REFERENCE_FACE_DISTANCE_SLIDER:
        REFERENCE_FACE_DISTANCE_SLIDER.change(
            face_selector.update_reference_face_distance,
            inputs=REFERENCE_FACE_DISTANCE_SLIDER,
        )
