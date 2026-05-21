from typing import Optional

import gradio

from facefusion import state_manager

PREVIEW_UPDATE_SECONDS_SLIDER: Optional[gradio.Slider] = None


def render() -> None:
    global PREVIEW_UPDATE_SECONDS_SLIDER
    seconds = state_manager.get_item('preview_update_seconds')
    if seconds is None:
        seconds = 2.0
    PREVIEW_UPDATE_SECONDS_SLIDER = gradio.Slider(
        label='Live preview update interval (seconds of footage)',
        value=float(seconds),
        minimum=0.5,
        maximum=10.0,
        step=0.5,
    )


def listen() -> None:
    PREVIEW_UPDATE_SECONDS_SLIDER.release(
        update_preview_update_seconds,
        inputs=[PREVIEW_UPDATE_SECONDS_SLIDER],
    )


def update_preview_update_seconds(seconds: float) -> None:
    state_manager.set_item('preview_update_seconds', float(seconds))
