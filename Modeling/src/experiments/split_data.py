import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def get_participant_data(root_dir):
    # Load and preprocess data
    data = []
    root = Path(root_dir)
    for participant_dir in root.glob("*"):
        if not participant_dir.is_dir():
            continue
        participant_id = participant_dir.name
        for video_file in participant_dir.glob("OBS/*.mp4"):
            filename_parts = video_file.stem.split("_")
            label = int(filename_parts[2])
            # if label != 2:  # Ignore neutral class
            binary_label = 1 if label > 2 else 0
            data.append(
                {
                    "ParticipantId": participant_id,
                    "Label": binary_label,
                }
            )
    return pd.DataFrame(data)


def format_split(df, split):
    return df[["ParticipantId"]].drop_duplicates().assign(Split=split)


def holdout_a_set(data: pd.DataFrame, holdout_set, tag="holdout"):
    holdout_set = set(holdout_set)
    mask = data.ParticipantId.isin(holdout_set)
    holdout_df = format_split(data[mask], tag)
    new_data = data[~mask]
    return new_data, holdout_df


def generate_inner_folds(dev_df: pd.DataFrame, n_val_splits=7, random_state=42, apply_holdout=False):
    val_sgkf = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=random_state)
    holdout_df = None
    if apply_holdout:
        for train_index, val_index in val_sgkf.split(dev_df, dev_df.Label, dev_df.ParticipantId):
            holdout_set = set(dev_df.iloc[val_index].ParticipantId.unique())
            dev_df, holdout_df = holdout_a_set(dev_df, holdout_set, tag="holdout")
            break
    inner_folds = []
    for train_index, val_index in val_sgkf.split(dev_df, dev_df.Label, dev_df.ParticipantId):
        train_df = format_split(dev_df.iloc[train_index], "train")
        val_df = format_split(dev_df.iloc[val_index], "validation")
        result = [train_df, val_df]
        if apply_holdout:
            result.append(holdout_df)
        inner_folds.append(pd.concat(result, ignore_index=True))
    return inner_folds


def save_inner_folds(inner_folds: list, root: Path, test_df=None):
    for j, fold in enumerate(inner_folds):
        if test_df is not None:
            fold = pd.concat((fold, test_df), ignore_index=True)
        fold.to_csv(root / f"{j}.csv", index=None)


def generate_splits(participant_data: pd.DataFrame, out_root: Path, n_test_splits=7, n_val_splits=7, random_state=42,
                    holdout_test_set: set = None, holdout_val: bool = False
                    ):
    if holdout_test_set:
        participant_data, test_df = holdout_a_set(participant_data, holdout_test_set, tag="test")
        inner_folds = generate_inner_folds(participant_data, n_val_splits, random_state, holdout_val)

        n_test_splits -= 1
        os.makedirs(out_root / f"fold_{n_test_splits}#HP", exist_ok=True)
        save_inner_folds(inner_folds, out_root / f"fold_{n_test_splits}#HP", test_df=test_df)

    test_sgkf = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=random_state)

    for i, (dev_index, test_index) in enumerate(
            test_sgkf.split(participant_data, participant_data.Label, participant_data.ParticipantId)):
        os.makedirs(out_root / f"fold_{i}", exist_ok=True)
        test_df = format_split(participant_data.iloc[test_index], "test")
        dev_df = participant_data.iloc[dev_index]

        inner_folds = generate_inner_folds(dev_df, n_val_splits, random_state, holdout_val)
        save_inner_folds(inner_folds, out_root / f"fold_{i}", test_df=test_df)


def main():
    parser = argparse.ArgumentParser(description='Generate nested CV splits')
    parser.add_argument('--root_dir', required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for split files')
    parser.add_argument('--n_test_splits', type=int, default=7,
                        help='Number of test splits')
    parser.add_argument('--n_val_splits', type=int, default=7,
                        help='Number of validation splits')
    parser.add_argument('--holdout_test_set', nargs='+',
                        help='Participants to use as test set in one outer fold')
    parser.add_argument("--holdout_val", action="store_true", default=False,
                        help="Holdout one validation set for ensemble evaluation")
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    # Create nested splits
    generate_splits(
        participant_data=get_participant_data(Path(args.root_dir)),
        out_root=Path(args.output_dir),
        n_test_splits=args.n_test_splits,
        n_val_splits=args.n_val_splits,
        random_state=args.random_state,
        holdout_test_set=args.holdout_test_set,
        holdout_val=args.holdout_val
    )


if __name__ == '__main__':
    main()
