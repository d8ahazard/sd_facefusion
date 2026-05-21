from typing import Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.user_data import load_ui_defaults, save_ui_defaults
from facefusion.uis.core import get_reload_outputs, reload_all_settings, register_ui_component

SET_DEFAULT_BUTTON: Optional[gradio.Button] = None
RESET_DEFAULT_BUTTON: Optional[gradio.Button] = None
DEFAULTS_STATUS: Optional[gradio.Textbox] = None


def render() -> None:
    global SET_DEFAULT_BUTTON
    global RESET_DEFAULT_BUTTON
    global DEFAULTS_STATUS

    with gradio.Row():
        SET_DEFAULT_BUTTON = gradio.Button('Set as default', variant='primary', size='sm')
        RESET_DEFAULT_BUTTON = gradio.Button('Reset to saved default', size='sm')
    DEFAULTS_STATUS = gradio.Textbox(
        label='Defaults',
        value='Settings load from user_data/ui_defaults.json on startup when present.',
        interactive=False,
        max_lines=2,
    )
    register_ui_component('settings_defaults_status', DEFAULTS_STATUS)


def listen() -> None:
    reload_outputs = get_reload_outputs()

    def set_default() -> str:
        if save_ui_defaults():
            return 'Saved current settings as default (applied on next WebUI launch).'
        return 'Failed to save defaults.'

    def _auto_padding_reset_updates():
        from facefusion.uis.components import face_masker

        model = state_manager.get_item('auto_padding_model') or 'None'
        return face_masker.update_auto_padding_model_and_ui(model)

    def reset_default():
        if load_ui_defaults():
            msg = 'Loaded saved default settings.'
        else:
            msg = 'No saved defaults found at user_data/ui_defaults.json.'
        if not reload_outputs:
            return (msg,)
        return (msg, *reload_all_settings(), *_auto_padding_reset_updates())

    def _reset_outputs():
        from facefusion.uis.components import face_masker

        outputs = [DEFAULTS_STATUS] + list(reload_outputs or [])
        for component in (
            face_masker.AUTO_PADDING_STATUS,
            face_masker.AUTO_PADDING_CONFIDENCE_SLIDER,
            face_masker.AUTO_PADDING_INTERSECTION_THRESHOLD_SLIDER,
            face_masker.AUTO_PADDING_MASK_AREAS_CHECKBOX_GROUP,
            face_masker.MASK_ENABLE_BUTTON,
            face_masker.MASK_DISABLE_BUTTON,
            face_masker.MASK_CLEAR_BUTTON,
        ):
            if component is not None and component not in outputs:
                outputs.append(component)
        return outputs

    SET_DEFAULT_BUTTON.click(set_default, outputs=[DEFAULTS_STATUS])
    reset_outputs = _reset_outputs()
    if len(reset_outputs) > 1:
        RESET_DEFAULT_BUTTON.click(reset_default, outputs=reset_outputs)
    else:
        RESET_DEFAULT_BUTTON.click(lambda: reset_default()[0], outputs=[DEFAULTS_STATUS])
