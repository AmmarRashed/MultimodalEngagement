"""
These functions are the main components of the feature extraction module for the Webcam clips.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from moviepy.video.fx.mirror_x import mirror_x
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.spatial import ConvexHull
from tqdm.notebook import tqdm

from feature_extraction.emonet import EmoNet
from feature_extraction.head_pose_sixdrepnet import HeadPoseModel


def save_into_file(frames, output_path):
    clip = ImageSequenceClip(list(frames), fps=30)
    clip.write_videofile(output_path, verbose=False,
                         audio=False, logger=None, fps=30)
    clip.close()


def pillowfy_frames(frames):
    return (frames * 255).cpu().byte().permute(0, 2, 3, 1).numpy()


def parse_video(path, torch_ready=True, sample_rate=1, max_len=60, mirror=False):
    clip = VideoFileClip(path).resize(height=256)
    if clip.duration > max_len:
        clip = clip.subclip(t_start=-max_len)
    if mirror:
        clip = mirror_x(clip)
    frames = clip.iter_frames()
    frames = np.stack([f for i, f in enumerate(frames) if i % sample_rate == 0])
    clip.close()
    if not torch_ready:
        return frames

    return torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255


def get_landmarks_from_heatmap(heatmap, shape):
    landmarks = []
    w, h = shape
    heatmap_shape = heatmap.shape[1:]
    for i in range(heatmap.shape[0]):
        split = heatmap[i, :, :]
        max_index = np.argmax(split, axis=None)
        y, x = np.unravel_index(max_index, split.shape)
        x = int(x * w / heatmap_shape[0])
        y = int(y * h / heatmap_shape[1])
        landmarks.append((x, y))
    return np.array(landmarks)


def get_bounding_box(facial_landmarks, margin=0.3, nose_center=False):
    min_x = min(facial_landmarks[:, 0])
    max_x = max(facial_landmarks[:, 0])
    min_y = min(facial_landmarks[:, 1])
    max_y = max(facial_landmarks[:, 1])

    width = abs(max_x - min_x)
    height = abs(max_y - min_y)

    side_length = int(max(width, height) * (1 + margin))
    if nose_center:
        center_x, center_y = facial_landmarks[30]  # nose tip
    else:
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
    half_side = round(side_length / 2)
    top_left_x = center_x - half_side
    top_left_y = center_y - half_side
    bottom_right_x = center_x + half_side
    bottom_right_y = center_y + half_side

    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def crop_face_with_landmarks(image, landmarks, plot=True):
    # Compute the convex hull of the landmarks
    hull = ConvexHull(landmarks)

    if plot:
        # Plot the convex hull
        fig, ax = plt.subplots(ncols=2)
        ax[0].axis('off')
        ax[0].imshow(image)
        ax[0].scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='o', s=5)  # plot landmarks
        for simplex in hull.simplices:
            ax[0].plot(landmarks[simplex, 0], landmarks[simplex, 1], 'k-')

    # Create a black image (mask) with the same shape as the original image
    mask = np.zeros_like(image)

    # Draw the convex hull on the mask
    cv2.fillPoly(mask, [(landmarks[hull.vertices]).astype(np.int32)], color=(255, 255, 255))
    # Apply the mask to the original image using bitwise_and
    result = cv2.bitwise_and(image, mask)
    if plot:
        ax[1].scatter(landmarks[hull.vertices, 0], landmarks[hull.vertices, 1], c='r', marker='o', s=5)

        ax[1].imshow(result)
        ax[1].axis('off');
    return result


def add_forehead_margin(landmarks, margin=20):
    # Get the indices of the points above the eyebrows landmarks (i.e. 17 for left and 26 for right)
    left_eyebrow_y = landmarks[17][1]
    right_eyebrow_y = landmarks[26][1]
    max_y_pos = max(left_eyebrow_y, right_eyebrow_y)  # bottom-most eyebrow landmark
    indices_above_eyebrow = np.where(landmarks[:, 1] <= (max_y_pos))[0]
    landmarks_with_margin = landmarks.copy()
    landmarks_with_margin[indices_above_eyebrow, 1] -= margin
    return landmarks_with_margin


def crop_bounding_box(image, box, resize=True):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = box
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    if resize:
        orig_size = image.shape[:-1]
        cropped_image = cv2.resize(cropped_image, orig_size)
    return cropped_image


class WebcamFeatureExtractionModule:
    def __init__(self, emonet_path, sixdrepnet_path, n_expression=8, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emonet_path = emonet_path
        self.sixdrepnet_path = sixdrepnet_path
        self.n_expression = n_expression
        self.device = device

        self.emonet = EmoNet(n_expression=self.n_expression).to(self.device)
        self.load_emonet_weights()

        self.pose_model = HeadPoseModel(self.sixdrepnet_path, self.device)

    def load_emonet_weights(self):
        state_dict = torch.load(self.emonet_path, map_location='cpu')
        self.emonet.load_state_dict(state_dict, strict=True)
        for param in self.emonet.parameters():
            param.requires_grad = False
        self.emonet.eval()

    def process_frames_batch(self, frames, forehead_margin=20):
        with torch.no_grad():
            out = self.emonet(frames.to(self.device))
        frames = pillowfy_frames(frames)
        size = frames.shape[1:-1]
        faces = []
        face_masks = []
        batch_landmarks = []
        filtered_frames = []
        indices = []
        for i, (h, f) in tqdm(enumerate(zip(out["heatmap"].cpu(), frames)), desc="face", leave=False):
            landmarks = get_landmarks_from_heatmap(h, size)
            try:
                # face
                landmarks_with_margin = add_forehead_margin(landmarks, 30)
                face_square = crop_bounding_box(f, get_bounding_box(landmarks_with_margin, margin=0.1))

                # face mask
                landmarks_with_margin = add_forehead_margin(landmarks, forehead_margin)
                face_mask = crop_face_with_landmarks(f, landmarks_with_margin, plot=False)
                facemask_square = crop_bounding_box(face_mask,
                                                    get_bounding_box(landmarks_with_margin, margin=0, nose_center=True))
            except cv2.error:
                continue
            faces.append(face_square)
            face_masks.append(facemask_square)
            batch_landmarks.append(landmarks)
            filtered_frames.append(f)
            indices.append(i)

        # recalculate face embeddings with face-only input
        faces = torch.tensor(np.array(faces)).permute(0, 3, 1, 2) / 255
        with torch.no_grad():
            out = self.emonet(faces.to(self.device))
        head_pose_features = self.pose_model.get_head_pose_features(filtered_frames, batch_landmarks)
        return np.array(face_masks), head_pose_features, out, np.array(batch_landmarks), np.array(indices)

    def process_video(self, video_path, sample_rate=1, max_batch_size=50, forehead_margin=20, mirror=False):
        frames = parse_video(video_path, sample_rate=sample_rate, mirror=mirror)
        num_batches = len(frames) // max_batch_size
        batches = np.array_split(frames, max(num_batches, 1))
        faces = []
        features = []
        pose_features = []
        landmarks = []
        frame_indices = []
        accumulator = 0  # not to lose the global index of frames after batching and filtering.
        for batch in tqdm(batches, desc=os.path.basename(video_path), leave=False):
            try:
                batch_faces, batch_head_poses, batch_out, batch_landmarks, batch_indices = self.process_frames_batch(
                    batch, forehead_margin=forehead_margin)
                batch_indices += accumulator
                accumulator += len(batch_faces)
            except:
                continue
            faces.extend(batch_faces)
            pose_features.extend(batch_head_poses)
            landmarks.extend(batch_landmarks)
            frame_indices.extend(batch_indices)
            features.extend(
                torch.cat(
                    (batch_out['embedding'], batch_out['expression'],
                     batch_out['valence'].unsqueeze(1), batch_out['arousal'].unsqueeze(1)),
                    dim=1).cpu().numpy())
        if len(pose_features) != len(features):
            print(
                f"{os.path.basename(video_path)}: Head pose features ({len(pose_features)})!= other features ({len(features)})")
        return np.array(faces), np.array(features), torch.stack(pose_features).numpy(), np.array(landmarks), np.array(
            frame_indices)
