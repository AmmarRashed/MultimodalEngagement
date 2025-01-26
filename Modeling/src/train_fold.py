# train_fold.py
import argparse
import os
import sys
from pathlib import Path
from typing import Dict

from experiments.evaluation_run import EvaluationRun
from utils import Paths, read_yaml_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a model on a specific fold of data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "fold",
        type=int,
        help="Fold number to process"
    )

    parser.add_argument(
        "--paths-config",
        type=Path,
        required=True,
        help="Path to the paths.json configuration file"
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        required=True,
        help="Path to the training configurations YAML file"
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory containing the data splits"
    )

    return parser.parse_args()


def setup_training(args: argparse.Namespace) -> tuple[Dict, Path]:
    """Set up the training configuration and paths.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (configs dict, split path)
    """
    paths = Paths(args.paths_config)
    configs = read_yaml_config(args.training_config)

    # Update configs with paths
    configs["root"] = paths["FeaturesRoot"]
    configs["mirrored_root"] = paths["MirroredFeaturesRoot"]
    configs["runs_path"] = paths["RunsPath"]

    # Construct split path
    split_path = args.splits_dir / f"{args.fold}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    return configs, paths


def main():
    """Main execution function."""
    args = parse_arguments()

    # try:
    configs, paths = setup_training(args)

    # Setup for specific fold
    configs["splits_path"] = str(args.splits_dir / f"{args.fold}.csv")

    eval_run = EvaluationRun(
        configs=configs,
        estimator_name="Transformer",
        targets=["engagement"],
        exp_name="TransformerEngagement",
        exp_tag=f"all_data#{args.fold}",
        output_path=os.path.join(
            paths["TrainedModels"], f"TransformerEngagement_{args.fold}"
        ),
        verbose=False
    )
    eval_run.run()

    # except Exception as e:
    #     print(f"Error processing fold {args.fold}: {str(e)}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()
