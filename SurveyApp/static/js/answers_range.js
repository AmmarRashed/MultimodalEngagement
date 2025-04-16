const sliders = document.querySelectorAll(".question")
var selected_labels = {};

function highlight(widget) {
    widget.style.transform = `scale(1.3)`;
    // widget.style.fontWeight = 'bold';
    // widget.style.textShadow = "1px 1px 2px #333";
}

function reset_highlight(widget) {
    widget.style.transform = `scale(1)`;
    // widget.style.fontWeight = 'normal';
    widget.style.textShadow = null;
}

sliders.forEach((slider) => {
    // Find the associated value div using the data-for attribute
    const default_label = document.getElementById(slider.id + "-choice-0")
    highlight(default_label);
    selected_labels[slider.id] = default_label;

    slider.addEventListener("input", (event) => {
        const tempSliderValue = event.target.value;
        const chosen = document.getElementById(slider.id + "-choice-" + tempSliderValue)
        reset_highlight(selected_labels[slider.id]);
        highlight(chosen);
        selected_labels[slider.id] = chosen;
        const progress = (tempSliderValue / slider.max) * 100;

        slider.style.background = `linear-gradient(to right, #0dcaf0 ${progress}%, #ccc ${progress}%)`;
    })
})