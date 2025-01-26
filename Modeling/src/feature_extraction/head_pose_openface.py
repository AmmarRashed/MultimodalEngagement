import os
import subprocess
from pathlib import Path
from typing import List

from joblib import Parallel, delayed
from tqdm import tqdm

from utils import Paths

"""
Ensure that you include src/ in sys.path ```sys.path.append("<path to src>")```
"""


def validate_paths(paths: Paths) -> None:
    """Validate that all required paths exist."""
    if not paths.OpenFaceBin.exists():
        raise FileNotFoundError(f"OpenFace binary not found at: {paths.OpenFaceBin}")
    if not paths.Videos.exists():
        raise FileNotFoundError(f"Input directory not found at: {paths.Videos}")
    paths.OpenFaceOutput.mkdir(parents=True, exist_ok=True)


def get_command(paths: Paths, video_path: Path) -> List[str]:
    """Generate the OpenFace command for feature extraction."""
    return [
        str(paths.OpenFaceBin),
        "-f", str(video_path),
        "-pose",
        "-aus",
        "-gaze",
        "-out_dir", str(paths.OpenFaceOutput)
    ]


def process_video(paths: Paths, video_path: Path) -> None:
    """Process a single video file using OpenFace."""
    command = get_command(paths, video_path)
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing video {video_path}: {e}")


def is_valid_folder(folder_name: str) -> bool:
    """Check if the folder should be processed."""
    return not folder_name.startswith(".") and folder_name.isnumeric()


def main():
    """Main execution function."""
    try:
        # Load paths
        paths = Paths('config/paths.json')

        # Validate paths
        validate_paths(paths)

        for root, _, files in tqdm(list(os.walk(paths.Videos))):
            folder = Path(root).name
            if not is_valid_folder(folder):
                continue

            video_paths = [Path(root) / f for f in files]

            Parallel(n_jobs=10)(  # You might want to make this configurable
                delayed(process_video)(paths, video_path)
                for video_path in tqdm(
                    video_paths,
                    leave=False,
                    desc=folder
                )
            )

    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
