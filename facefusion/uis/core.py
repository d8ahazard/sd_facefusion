import importlib
import os
import warnings
from types import ModuleType
from typing import Any, Dict, List, Optional

import gradio
from gradio.themes import Size

from facefusion import logger, metadata, state_manager, wording
from facefusion.exit_helper import hard_exit
from facefusion.filesystem import resolve_relative_path
from facefusion.uis import overrides
from facefusion.uis.typing import Component, ComponentName

os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

warnings.filterwarnings('ignore', category=UserWarning, module='gradio')

gradio.processing_utils.encode_array_to_base64 = overrides.encode_array_to_base64
gradio.processing_utils.encode_pil_to_base64 = overrides.encode_pil_to_base64

UI_COMPONENTS: Dict[ComponentName, Component] = {}
UI_LAYOUT_MODULES: List[ModuleType] = []
UI_LAYOUT_METHODS = \
    [
        'pre_check',
        'render',
        'listen',
        'run'
    ]

# Mapping of UI component names to their state_manager keys for automatic reload
COMPONENT_STATE_KEYS: Dict[str, str] = {
    # Processors
    'processors_checkbox_group': 'processors',
    # Face Swapper
    'face_swapper_model_dropdown': 'face_swapper_model',
    'face_swapper_pixel_boost_dropdown': 'face_swapper_pixel_boost',
    # Face Detector
    'face_detector_model_dropdown': 'face_detector_model',
    'face_detector_size_dropdown': 'face_detector_size',
    'face_detector_angles_checkbox_group': 'face_detector_angles',
    'face_detector_score_slider': 'face_detector_score',
    # Face Mask
    'face_mask_types_checkbox_group': 'face_mask_types',
    'face_mask_regions_checkbox_group': 'face_mask_regions',
    'face_mask_blur_slider': 'face_mask_blur',
    'auto_padding_model_dropdown': 'auto_padding_model',
    'auto_padding_confidence_slider': 'auto_padding_confidence',
    'auto_padding_intersection_threshold_slider': 'auto_padding_intersection_threshold',
    # Face Selector
    'face_selector_mode_dropdown': 'face_selector_mode',
    'face_selector_order_dropdown': 'face_selector_order',
    'face_selector_gender_dropdown': 'face_selector_gender',
    'face_selector_race_dropdown': 'face_selector_race',
    'reference_face_distance_slider': 'reference_face_distance',
    # Face Enhancer
    'face_enhancer_model_dropdown': 'face_enhancer_model',
    'face_enhancer_blend_slider': 'face_enhancer_blend',
    # Frame Enhancer
    'frame_enhancer_model_dropdown': 'frame_enhancer_model',
    'frame_enhancer_blend_slider': 'frame_enhancer_blend',
    # Execution
    'execution_thread_count_slider': 'execution_thread_count',
    'execution_queue_count_slider': 'execution_queue_count',
    # Face Landmarker
    'face_landmarker_model_dropdown': 'face_landmarker_model',
    'face_landmarker_score_slider': 'face_landmarker_score',
    # Frame Colorizer
    'frame_colorizer_model_dropdown': 'frame_colorizer_model',
    'frame_colorizer_blend_slider': 'frame_colorizer_blend',
    'frame_colorizer_size_dropdown': 'frame_colorizer_size',
    # Lip Syncer
    'lip_syncer_model_dropdown': 'lip_syncer_model',
    # Age Modifier
    'age_modifier_model_dropdown': 'age_modifier_model',
    'age_modifier_direction_slider': 'age_modifier_direction',
    # Expression Restorer
    'expression_restorer_model_dropdown': 'expression_restorer_model',
    'expression_restorer_factor_slider': 'expression_restorer_factor',
    # Face Editor
    'face_editor_model_dropdown': 'face_editor_model',
    # Style Changer
    'style_changer_model_dropdown': 'style_changer_model',
    # UI Workflow
    'ui_workflow_dropdown': 'ui_workflow',
}

# Components that need special handling (tuples, etc.)
COMPONENT_SPECIAL_KEYS: Dict[str, tuple] = {
    'face_mask_padding_top_slider': ('face_mask_padding', 0),
    'face_mask_padding_right_slider': ('face_mask_padding', 1),
    'face_mask_padding_bottom_slider': ('face_mask_padding', 2),
    'face_mask_padding_left_slider': ('face_mask_padding', 3),
}


def load_ui_layout_module(ui_layout: str) -> Any:
    try:
        ui_layout_module = importlib.import_module('facefusion.uis.layouts.' + ui_layout)
        for method_name in UI_LAYOUT_METHODS:
            if not hasattr(ui_layout_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError as exception:
        logger.error(wording.get('ui_layout_not_loaded').format(ui_layout=ui_layout), __name__)
        logger.debug(exception.msg, __name__)
        hard_exit(1)
    except NotImplementedError:
        logger.error(wording.get('ui_layout_not_implemented').format(ui_layout=ui_layout), __name__)
        hard_exit(1)
    return ui_layout_module


def get_ui_layouts_modules(ui_layouts: List[str]) -> List[ModuleType]:
    global UI_LAYOUT_MODULES

    if not UI_LAYOUT_MODULES:
        for ui_layout in ui_layouts:
            ui_layout_module = load_ui_layout_module(ui_layout)
            UI_LAYOUT_MODULES.append(ui_layout_module)
    return UI_LAYOUT_MODULES


def get_ui_component(component_name: ComponentName) -> Optional[Component]:
    if component_name in UI_COMPONENTS:
        return UI_COMPONENTS[component_name]
    return None


def get_ui_components(component_names: List[ComponentName]) -> Optional[List[Component]]:
    ui_components = []

    for component_name in component_names:
        component = get_ui_component(component_name)
        if component:
            ui_components.append(component)
    return ui_components


def register_ui_component(component_name: ComponentName, component: Component) -> None:
    component_elem_id = "ff3_" + component_name
    if component_name not in UI_COMPONENTS:
        try:
            setattr(component, 'elem_id', component_elem_id)
            setattr(component, 'do_not_save_to_config', True)
        except AttributeError:
            component.elem_id = component_elem_id
        UI_COMPONENTS[component_name] = component


def get_valid_reload_components() -> List[tuple]:
    """
    Get list of (component, state_key, is_special, index) tuples for valid registered components.
    Must be called AFTER all components are registered (after render() calls).
    """
    valid_components = []
    
    # Standard components
    for component_name in COMPONENT_STATE_KEYS:
        component = get_ui_component(component_name)
        if component is not None:
            state_key = COMPONENT_STATE_KEYS[component_name]
            valid_components.append((component, state_key, False, None))
    
    # Special components (tuple values)
    for component_name in COMPONENT_SPECIAL_KEYS:
        component = get_ui_component(component_name)
        if component is not None:
            state_key, index = COMPONENT_SPECIAL_KEYS[component_name]
            valid_components.append((component, state_key, True, index))
    
    return valid_components


def reload_all_settings() -> tuple:
    """
    Reload all UI component values from state_manager.
    Called automatically on page load/reconnect via gr.Blocks.load event.
    """
    valid_components = get_valid_reload_components()
    updates = []
    
    for component, state_key, is_special, index in valid_components:
        value = state_manager.get_item(state_key)
        
        if is_special:
            # Handle tuple values (like face_mask_padding)
            if value is not None and isinstance(value, (list, tuple)) and len(value) > index:
                updates.append(gradio.update(value=value[index]))
            else:
                updates.append(gradio.update())
        else:
            # Standard values
            if value is not None:
                updates.append(gradio.update(value=value))
            else:
                updates.append(gradio.update())
    
    return tuple(updates)


def get_reload_outputs() -> List[Component]:
    """
    Get list of UI components that will be updated by reload_all_settings().
    Must be called AFTER all components are registered (after render() calls).
    Only returns non-None components to avoid Gradio errors.
    """
    valid_components = get_valid_reload_components()
    return [component for component, _, _, _ in valid_components]


def launch() -> None:
    ui_layouts_total = len(state_manager.get_item('ui_layouts'))
    with gradio.Blocks(theme=get_theme(), css=get_css(), title=metadata.get('name') + ' ' + metadata.get('version'),
                       fill_width=True) as ui:
        for ui_layout in state_manager.get_item('ui_layouts'):
            ui_layout_module = load_ui_layout_module(ui_layout)

            if ui_layouts_total > 1:
                with gradio.Tab(ui_layout):
                    ui_layout_module.render()
                    ui_layout_module.listen()
            else:
                ui_layout_module.render()
                ui_layout_module.listen()

    for ui_layout in state_manager.get_item('ui_layouts'):
        ui_layout_module = load_ui_layout_module(ui_layout)
        ui_layout_module.run(ui)


def get_theme() -> gradio.Theme:
    return gradio.themes.Base(
        primary_hue=gradio.themes.colors.red,
        secondary_hue=gradio.themes.colors.neutral,
        radius_size=Size(
            xxs='0.375rem',
            xs='0.375rem',
            sm='0.375rem',
            md='0.375rem',
            lg='0.375rem',
            xl='0.375rem',
            xxl='0.375rem',
        ),
        font=gradio.themes.GoogleFont('Open Sans')
    ).set(
        background_fill_primary='*neutral_100',
        block_background_fill='white',
        block_border_width='0',
        block_label_background_fill='*neutral_100',
        block_label_background_fill_dark='*neutral_700',
        block_label_border_width='none',
        block_label_margin='0.5rem',
        block_label_radius='*radius_md',
        block_label_text_color='*neutral_700',
        block_label_text_size='*text_sm',
        block_label_text_color_dark='white',
        block_label_text_weight='600',
        block_title_background_fill='*neutral_100',
        block_title_background_fill_dark='*neutral_700',
        block_title_padding='*block_label_padding',
        block_title_radius='*block_label_radius',
        block_title_text_color='*neutral_700',
        block_title_text_size='*text_sm',
        block_title_text_weight='600',
        block_padding='0.5rem',
        border_color_primary='transparent',
        border_color_primary_dark='transparent',
        button_large_padding='2rem 0.5rem',
        button_large_text_weight='normal',
        button_primary_background_fill='*primary_500',
        button_primary_text_color='white',
        button_secondary_background_fill='white',
        button_secondary_border_color='transparent',
        button_secondary_border_color_dark='transparent',
        button_secondary_border_color_hover='transparent',
        button_secondary_border_color_hover_dark='transparent',
        button_secondary_text_color='*neutral_800',
        button_small_padding='0.75rem',
        checkbox_background_color='*neutral_200',
        checkbox_background_color_selected='*primary_600',
        checkbox_background_color_selected_dark='*primary_700',
        checkbox_border_color_focus='*primary_500',
        checkbox_border_color_focus_dark='*primary_600',
        checkbox_border_color_selected='*primary_600',
        checkbox_border_color_selected_dark='*primary_700',
        checkbox_label_background_fill='*neutral_50',
        checkbox_label_background_fill_hover='*neutral_50',
        checkbox_label_background_fill_selected='*primary_500',
        checkbox_label_background_fill_selected_dark='*primary_600',
        checkbox_label_text_color_selected='white',
        input_background_fill='*neutral_50',
        shadow_drop='none',
        slider_color='*primary_500',
        slider_color_dark='*primary_600'
    )


def get_css() -> str:
    overrides_css_path = resolve_relative_path('uis/assets/overrides.css')
    return open(overrides_css_path, 'r').read()
