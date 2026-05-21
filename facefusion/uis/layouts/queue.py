import gradio

from facefusion import state_manager
from facefusion.uis.components import queue_dashboard


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    with gradio.Blocks() as layout:
        with gradio.Row():
            with gradio.Column(scale=7):
                queue_dashboard.render_preview()
            with gradio.Column(scale=5):
                queue_dashboard.render_tables_and_actions()
    return layout


def listen() -> None:
    queue_dashboard.listen()


def register_tab_select(queue_tab) -> None:
    if queue_tab is None or not hasattr(queue_tab, 'select'):
        return
    outputs = _get_tab_refresh_outputs()
    if outputs:
        queue_tab.select(
            fn=queue_dashboard._refresh_on_tab,
            outputs=outputs,
            show_progress='hidden',
        )
        # A1111 Gradio uses _js (not js) on tab.select — see modules/ui_extra_networks.py
        queue_tab.select(
            fn=None,
            _js='function(){on_queue_tab_visible();}',
            inputs=[],
            outputs=[],
            show_progress=False,
        )


def _get_tab_refresh_outputs() -> list:
    from facefusion.uis.components import queue_dashboard as qd

    outputs = []
    for component in [
        qd.ACTIVE_QUEUE_TABLE,
        qd.HISTORY_QUEUE_TABLE,
        qd.ACTIVE_JOB_IDS,
        qd.HISTORY_JOB_IDS,
        qd.JOB_PREVIEW_IMAGE,
        qd.JOB_MAPPING_GALLERY,
        qd.JOB_PROGRESS_HTML,
        qd.JOB_LIVE_PREVIEW,
        qd.RUN_ALL_BUTTON,
        qd.RUN_SELECTED_BUTTON,
        qd.STOP_ALL_BUTTON,
        qd.STOP_SELECTED_BUTTON,
        qd.REMOVE_ALL_BUTTON,
        qd.REMOVE_SELECTED_BUTTON,
        qd.SELECTED_JOB_ID,
    ]:
        if component is not None:
            outputs.append(component)
    return outputs


def run(ui: gradio.Blocks) -> None:
    ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser'))
