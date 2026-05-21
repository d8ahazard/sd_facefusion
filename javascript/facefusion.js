let ffProgressTimeout = null;
let stoppedCount = 0;

function get_poll_button() {
    return gradioApp().getElementById("ff_queue_poll")
        || gradioApp().getElementById("ff3_check_status");
}

function start_status() {
    stoppedCount = 0;
    ffProgressTimeout = setTimeout(recheck_status, 2000);
}

/** Queue tab opened: refresh once, then resume polling if a job is already running (e.g. Map Start). */
function on_queue_tab_visible() {
    setTimeout(function () {
        var btn = get_poll_button();
        if (btn) {
            btn.click();
        }
        setTimeout(function () {
            var el = gradioApp().getElementById("statusDiv");
            if (el && el.dataset.started === "true") {
                start_status();
            }
        }, 400);
    }, 100);
}

function stop_status() {
    clearTimeout(ffProgressTimeout);
}

function recheck_status() {
    let status_element = gradioApp().getElementById("statusDiv");
    if (!status_element) {
        console.log("Can't find the status element.");
        return;
    }
    let btn = get_poll_button();
    if (!btn) {
        console.log("Can't find the poll button.");
        return;
    }
    btn.click();
    let started = status_element.dataset.started === 'true';
    if (!started) {
        ffProgressTimeout = setTimeout(recheck_status, 2000);
    } else {
        await_progress_stop();
    }
}

function await_progress_stop() {
    clearTimeout(ffProgressTimeout);
    let status_element = gradioApp().getElementById("statusDiv");
    if (!status_element) {
        console.log("Can't find the status element.");
        return;
    }
    console.log("Data-started value: ", status_element.dataset.started);
    let started = status_element.dataset.started === 'true';
    if (started) {
        let btn = get_poll_button();
        if (!btn) {
            console.log("Can't find the poll button.");
            return;
        }
        btn.click();
        ffProgressTimeout = setTimeout(await_progress_stop, 1000);
    } else {
        console.log("Job stopped.");
        let reloadBtn = gradioApp().getElementById("ff_settings_reload");
        if (reloadBtn) {
            reloadBtn.click();
        }
    }
}

function get_selected_row() {
    console.log("Incoming arguments:", arguments);
    let selected = document.querySelector(".selectRow.selected");
    let res = [-1, -1];
    if (selected) {
        let rId = selected.id.replace("row", "");
        res = [rId, rId];
    }
    console.log("Selected row: ", res);
    return res;
}

document.addEventListener("DOMContentLoaded", function () {
    document.addEventListener("click", function (e) {
        if (e.target.tagName === "TD") {
            if (e.target.parentElement.classList.contains("selectRow")) {
                let rows = document.querySelectorAll(".selectRow");
                let selected = e.target.parentElement.classList.contains("selected");
                for (let i = 0; i < rows.length; i++) {
                    rows[i].classList.remove("selected");
                }
                if (!selected) {
                    e.target.parentElement.classList.add("selected");
                    let ff3_toggle_remove = gradioApp().getElementById("ff3_toggle_remove");
                    if (ff3_toggle_remove) {
                        ff3_toggle_remove.click();
                    } else {
                        console.log("Can't find toggle remove");
                    }
                }
            }
        }
    });
});
