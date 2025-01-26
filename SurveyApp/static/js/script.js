var session_no_label = document.getElementById("session-no-label");
var game_selector = document.getElementById("game");
var game_pic = document.getElementById("game-pic");
var time_btn = document.getElementById("time-btn");
var start_ts_label = document.getElementById("start_ts");
var start_ts_reset = document.getElementById("reset-start");
var end_ts_label = document.getElementById("end_ts");
var end_ts_reset = document.getElementById("reset-end");
const level_selector = document.getElementById("game_level")
var started = false;
var finished = false;


// Start and End timers and Button
function reset_end() {
    finished = false;
    time_btn.textContent = "End";
    end_ts_reset.disabled = true;
    end_ts_label.value = null;
}

function reset_start() {
    reset_end();
    started = false;
    time_btn.textContent = "Start";
    start_ts_reset.disabled = true;
    start_ts_label.value = null;
}

start_ts_reset.addEventListener('click', reset_start);
end_ts_reset.addEventListener('click', reset_end);

function formatDateTime(dt) {
    const year = dt.getFullYear();
    const month = String(dt.getMonth() + 1).padStart(2, '0'); // Months are 0-indexed
    const day = String(dt.getDate()).padStart(2, '0');
    const hours = String(dt.getHours()).padStart(2, '0'); // 24-hour format
    const minutes = String(dt.getMinutes()).padStart(2, '0');
    const seconds = String(dt.getSeconds()).padStart(2, '0');

    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

time_btn.addEventListener('click', function () {
    var currentDateTime = new Date();
    var formatted_dt = formatDateTime(currentDateTime);
    if (!started) {
        started = true;
        console.log("Started game session at: " + formatted_dt);
        this.textContent = "End"
        start_ts_label.value = formatted_dt;
        start_ts_reset.disabled = false;

    } else if (started && !finished) {
        finished = true;
        console.log("Ended game session at: " + formatted_dt);
        this.textContent = "Submit"
        end_ts_label.value = formatted_dt;
        end_ts_reset.disabled = false;

    } else if (started && finished) {
        this.type = "submit";
    }
});

// Game Selection
function update_game_pic() {
    var src = game_selector.options[game_selector.selectedIndex].text.toLowerCase();
    if (game_selector.selectedIndex === 0)
        src = "select.svg"
    else
        src += ".png"
    game_pic.src = "/static/" + src;
}

function update_level_selector() {
    var game_id = game_selector.options[game_selector.selectedIndex].value;
    $.ajax(
        {
            url: get_selected_game_levels_url,
            method: 'GET',
            data: {
                game_id: game_id
            },
            success: function (response) {
                level_selector.innerHTML = '';
                response.levels.forEach(level => {
                    const option = document.createElement("option");
                    option.value = level[0];
                    option.textContent = level[1];
                    level_selector.appendChild(option);
                });
            }
        }
    );
    level_selector.selectedIndex = 0; // reset selection
}

update_game_pic();

// Session Management
function update_session(game_id, game_level) {
    $.ajax({
        url: update_session_url,
        method: 'GET',
        data: {
            game_id: game_id,
            game_level: game_level
        },
        success: function (response) {
            session_no_label.textContent = "Session: " + response.session_no;
        },
        error: function (xhr, status, error) {
            console.error('Error:', error);
        }
    });
}

$("#game").change(function () {
        update_game_pic();
        update_level_selector();
        update_session($("#game").val(), null);
    }
);
$("#game_level").change(function () {
    update_session($("#game").val(), $("#game_level").val());
});

// Answer buttons
var radioElements = document.querySelectorAll('input[type="radio"]');
radioElements.forEach(function (radio) {
    radio.classList.add('btn-check');
    var label = document.querySelector('label[for="' + radio.id + '"]');
    if (label) {
        label.classList.add("btn");
        label.classList.add("btn-outline-primary");
        label.classList.add("btn-survey");
    }

});
