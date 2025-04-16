import logging
import time
import argparse
from pynput import mouse

# time key pressed/released
time_str = time.strftime("%Y-%m-%d--%H-%M-%S")
logging.basicConfig(filename=f"{time_str}.mouse.tsv", level=logging.DEBUG, format='%(asctime)s\t%(message)s')


# Mouse
def on_move(x, y):
    logging.info(f"{x}\t{y}\tMove\t{None}\tMouse")


def on_move_disabled(x, y):
    pass


def on_click(x, y, button, pressed):
    logging.info(f'{x}\t{y}\t{"Pressed" if pressed else "Released"}\t{button}')


def on_scroll(x, y, dx, dy):
    logging.info(f'{x}\t{y}\tScroll\t{dx, dy}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--movement", dest='movement', action='store_true')
    args = parser.parse_args()
    with mouse.Listener(on_move=on_move if args.movement else on_move_disabled,
                        on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()
