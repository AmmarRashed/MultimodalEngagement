import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Paths

"""
Ensure that you include src/ in sys.path ```sys.path.append("<path to src>")```
"""


# OpenFace feature column definitions
class OpenFaceColumns:
    GAZE = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',
            'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
            'gaze_angle_x', 'gaze_angle_y']

    BLINK = ['AU45_r']

    AU_INTENSITIES = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
                      'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
                      'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r',
                      'AU26_r']

    AU_PRESENCE = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c',
                   'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c',
                   'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c',
                   'AU26_c', 'AU28_c', 'AU45_c']

    POSE = ['pose_Tx', 'pose_Ty', 'pose_Tz',
            'pose_Rx', 'pose_Ry', 'pose_Rz']

    @classmethod
    def get_feature_columns(cls) -> Dict[str, List[str]]:
        return {
            "gaze": cls.GAZE,
            "blink": cls.BLINK,
            "au_r": cls.AU_INTENSITIES,
            "au_c": cls.AU_PRESENCE,
            "pose": cls.POSE
        }


def validate_paths(paths: Paths) -> None:
    """Validate that all required paths exist."""
    if not paths.FeaturesRoot.exists():
        raise FileNotFoundError(f"Features directory not found at: {paths.FeaturesRoot}")
    if not paths.OpenFaceOutput.exists():
        raise FileNotFoundError(f"OpenFace output directory not found at: {paths.OpenFaceOutput}")


def is_valid_folder(folder_name: str) -> bool:
    """Check if the folder should be processed."""
    return not folder_name.startswith(".") and folder_name.isnumeric()


def merge_features(npz_path: Path, csv_path: Path, feature_columns: Dict[str, List[str]]) -> None:
    """Merge OpenFace features into existing NPZ file."""
    try:
        # Load existing features
        npz_file = dict(np.load(npz_path))

        # Load OpenFace features
        df = pd.read_csv(csv_path)

        # Add each feature group
        for fc, columns in feature_columns.items():
            npz_file[f"OpenFace_{fc}"] = df[columns].to_numpy()

        # Save updated features
        np.savez(npz_path, **npz_file)
    except Exception as e:
        print(f"Error processing file {npz_path}: {e}")


def main():
    """Main execution function."""
    try:
        # Load paths
        paths = Paths('config/paths.json')

        # Validate paths
        validate_paths(paths)

        # Get feature columns
        feature_columns = OpenFaceColumns.get_feature_columns()

        # Process files
        for root, _, files in tqdm(list(os.walk(paths.FeaturesRoot))):
            folder = Path(root).name
            if not is_valid_folder(folder):
                continue

            for f in tqdm(files, desc=folder, leave=False):
                if not f.endswith('.npz'):
                    continue

                npz_path = Path(root) / f
                csv_path = paths.OpenFaceOutput / f.replace("npz", 'csv')

                if not csv_path.exists():
                    print(f"Warning: OpenFace CSV file not found for {f}")
                    continue

                merge_features(npz_path, csv_path, feature_columns)

    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
