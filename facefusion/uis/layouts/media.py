import gradio

from facefusion import state_manager
from facefusion.uis.components import media_sources, media_targets, style_transfer_options


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    with gradio.Blocks() as layout:
        with gradio.Row():
            with gradio.Column(scale=5):
                gradio.Markdown('### Source people')
                media_sources.render()
            with gradio.Column(scale=5):
                gradio.Markdown('### Target media')
                media_targets.render()
        with gradio.Blocks():
            style_transfer_options.render()
    return layout


def listen() -> None:
    media_sources.listen()
    media_targets.listen()
    style_transfer_options.listen()


def run(ui: gradio.Blocks) -> None:
    ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser'))
