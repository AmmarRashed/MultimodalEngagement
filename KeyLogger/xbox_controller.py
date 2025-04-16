import inputs
import time
import logging

# Define the Xbox controller event types
EVENT_TYPES = {
    'ABS_X': 'left_stick_x',
    'ABS_Y': 'left_stick_y',
    'ABS_RX': 'right_stick_x',
    'ABS_RY': 'right_stick_y',
    'BTN_SOUTH': 'a_button',
    'BTN_EAST': 'b_button',
    'BTN_WEST': 'x_button',
    'BTN_NORTH': 'y_button',
    'BTN_TL': 'left_bumper',
    'BTN_TR': 'right_bumper',
    'BTN_THUMBL': 'left_stick_button',
    'BTN_THUMBR': 'right_stick_button',
    'ABS_HAT0X': 'dpad_x',
    'ABS_HAT0Y': 'dpad_y',
    'ABS_Z': 'left_trigger',
    'ABS_RZ': 'right_trigger',
    'BTN_SELECT': 'start_button',
    'BTN_START': 'back_button'
}

# time key pressed/released
time_str = time.strftime("%Y-%m-%d--%H-%M-%S")
logging.basicConfig(filename=f"{time_str}.xbox.tsv", level=logging.DEBUG, format='%(asctime)s\t%(message)s')


# Define a function to handle events
def handle_event(event):
    if event.code in EVENT_TYPES:
        logging.info(f"{EVENT_TYPES[event.code]}\t{event.ev_type}\t{event.state}")


while True:
    for event in inputs.get_gamepad():
        handle_event(event)
