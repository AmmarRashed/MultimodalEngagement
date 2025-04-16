import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sixdrepnet.model import SixDRepNet
from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices, plot_pose_cube
from torchvision.transforms import transforms


class HeadPoseModel(object):
    def __init__(self, model_path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        saved_state_dict = torch.load(model_path)
        self.model = SixDRepNet(backbone_name="RepVGG-B1g2", backbone_file='', deploy=True, pretrained=False)
        self.model.eval()
        self.model.load_state_dict(saved_state_dict)
        self.model.to(device)

        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

    def predict_rotation(self, frames):
        if len(frames) < 1:
            print("No frames to predict")
            return
        frames = torch.stack([self.transformations(TF.to_pil_image(f)) for f in frames]).to(self.device)
        with torch.no_grad():
            pred = self.model(frames)
        euler = compute_euler_angles_from_rotation_matrices(pred)
        return euler.cpu()

    @staticmethod
    def _calculate_translation(lm):
        x_min = min(lm[:, 0])
        x_max = max(lm[:, 0])
        y_min = min(lm[:, 1])
        y_max = max(lm[:, 1])
        width = abs(x_max - x_min)
        tdx = x_min + int(.5 * (x_max - x_min))
        tdy = y_min + int(.5 * (y_max - y_min))
        return tdx, tdy, width

    def _calculate_position(self, lm):
        x, y = lm[30]  # nose tip
        # depth proxy (side length of the face bounding box)
        # if the player leans forward (i.e. zoom) the bbox is larger, and vice versa
        _, _, z = self._calculate_translation(lm)
        return x, y, z

    def calculate_position(self, frames_landmarks):
        return torch.tensor(
            np.array([self._calculate_position(lm) for lm in frames_landmarks])
        )

    def get_head_pose_features(self, frames, landmarks):
        rotation = self.predict_rotation(frames)
        position = self.calculate_position(landmarks)
        return torch.cat((rotation, position), axis=-1)

    def _calculate_pnp(self, frame, landmarks):
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
        points = [
            30,  # nose tip
            8,  # chin
            36,  # left eye corner
            45,  # right eye corner
            48,  # left mouth corner
            54  # right mouth corner
        ]
        img_points = np.array(landmarks[points], dtype='double')

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, img_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        tx = translation_vector[:, 0]
        ty = translation_vector[:, 1]
        tz = translation_vector[:, 2]

        # get rotation matrix from rotation vector
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Pitch calculation
        pitch = np.arcsin(-rotation_matrix[2, 0])

        # Yaw calculation
        yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        # Roll calculation
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Convert radians to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return torch.tensor(np.array([pitch, yaw, roll, tx, ty, tz]))

    def _plot_pose_cube(self, img, rot_deg, lm):
        img = img.copy()
        p, y, r = rot_deg
        tdx, tdy, width = self._calculate_translation(lm)
        plot_pose_cube(img, y, p, r, tdx, tdy, width)
        return img

    def plot_pose_cube(self, frames, rotations, landmarks):
        deg = rotations * 180 / np.pi
        pitch = deg[:, 0]
        yaw = deg[:, 1]
        roll = deg[:, 2]

        images = []
        for f, lm, p, y, r in zip(frames, landmarks, pitch, yaw, roll):
            images.append(self._plot_pose_cube(f, (p, y, r), lm))

        return images
