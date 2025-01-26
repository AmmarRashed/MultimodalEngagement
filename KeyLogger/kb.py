import logging
import time

import keyboard
from keyboard import KEY_DOWN, KEY_UP

# time key pressed/released
time_str = time.strftime("%Y-%m-%d--%H-%M-%S")
logging.basicConfig(filename=f"{time_str}.kb.tsv", level=logging.DEBUG, format='%(asctime)s\t%(message)s')


def on_action(event):
    if event.event_type == KEY_DOWN:
        on_press(event.name)

    elif event.event_type == KEY_UP:
        on_release(event.name)


def on_press(key):
    logging.info(f'{key}\tPressed'.lower())


def on_release(key):
    logging.info(f'{key}\tReleased'.lower())


keyboard.hook(lambda e: on_action(e))

keyboard.wait()
