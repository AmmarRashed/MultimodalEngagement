"""
These functions are used to pre-process full participation recording (i.e. OBS scene)
into annotated webcam clips.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm.notebook import tqdm


def plot_frame(image_array: np.ndarray):
    plt.imshow(image_array)
    plt.axis("off")


def timestamps_to_time_offsets(start_ts, end_ts, creation_datetime):
    start_time = (start_ts - creation_datetime).total_seconds()
    end_time = (end_ts - creation_datetime).total_seconds()
    return start_time, end_time


def get_video_creation_datetime(path):
    date_time = os.path.split(path)[-1].split(".")[0]
    date, time = date_time.split(" ")
    datetime_str = f"{date} {time.replace('-', ':')}"
    creation_datetime = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S").tz_localize("US/Eastern")
    return creation_datetime


def extract_clip(video, session_info, creation_datetime, trim_start=2, trim_end=2):
    start_time, end_time = timestamps_to_time_offsets(
        session_info.start_ts, session_info.end_ts, creation_datetime)
    # the +2 and -2 accounts for the delay for logging the end ts.
    clip = video.subclip(start_time + trim_start, end_time - trim_end)
    return clip


def crop_with_ratio(clip, h_crop_ratio=[0.7, 0.0], w_crop_ratio=[0.0, 0.7]):
    """
    clip moviepy (editor) VideoFileClip object
    h_crop_ratio: list [from top, from bottom]
    w_crop_ratio: list [from left, from right]
    """
    w, h = clip.w, clip.h

    y1 = h * (1 - h_crop_ratio[0])
    y2 = h * (1 - h_crop_ratio[1])

    x1 = w_crop_ratio[0] * w
    x2 = w_crop_ratio[1] * w

    #     cropped_video = clip.crop(x1=0, y1=h - crop_height, x2=crop_width, y2=h)
    cropped_video = clip.crop(x1=x1, x2=x2, y1=y1, y2=y2)
    return cropped_video


def crop_square(clip, x1=120, x2=800, y1=270, y2=950):
    w, h = clip.w, clip.h
    # determined by exploring the data with the next cell
    return clip.crop(x1=x1, x2=x2, y1=y1, y2=y2)


def reduce_resolution(clip, target_height=480):
    return clip.resize(height=target_height)


def process_game_sessions(video_path, sessions_info, cropping_params=dict(x1=120, x2=800, y1=270, y2=950)):
    video = VideoFileClip(video_path)
    data = list()
    creation_datetime = get_video_creation_datetime(video_path)
    print(creation_datetime)
    for i, row in tqdm(sessions_info.iterrows(), desc="Gaming Session", leave=False):
        print(row.start_ts, row.end_ts)
        clip = extract_clip(video, row, creation_datetime)
        clip = crop_square(clip, **cropping_params)
        clip = reduce_resolution(clip)

        data.append({
            "clip": clip,
            "ans_id": row.id,
            "user_id": row.user_id,
            "engagement": row.engagement,
            "interest": row.interest,
            "stress": row.stress,
            "excitement": row.excitement
        })
    return data
