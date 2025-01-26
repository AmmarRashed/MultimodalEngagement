# optimize_hyperparams.py
import argparse
import json
import subprocess
from itertools import product
from pathlib import Path
from typing import Dict
from typing import Optional

import optuna
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from experiments.ensemble_evaluation import EnsembleEvaluation


def generate_augmentation_options():
    # Generate all possible augmentation combinations
    augs = ['mirror', 'shift']
    states = [True, False]
    combinations = []

    # Two augmentations combinations
    for combo in product(product([augs[0]], states), product([augs[1]], states)):
        combinations.append([list(item) for item in combo])

    # Single augmentation cases
    for aug in augs:
        for state in states:
            combinations.append([[aug, state]])

    # Add the no augmentation case
    combinations.append(None)
    return combinations


def suggest_hyperparameters(trial: optuna.Trial) -> Dict:
    """Suggest hyperparameters for this trial."""
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'decay': trial.suggest_float('decay', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'nheads': trial.suggest_categorical('nheads', [1, 2, 4]),
        'project_to': trial.suggest_categorical('project_to', [32, 64, 128]),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', .1, 0.7),
        'loss': 'bce_ordinal',
        'alpha': trial.suggest_float('alpha', .1, 1.5),
        'class_weights': True,  # trial.suggest_categorical('class_weights', [False, True])
        # 'gamma': trial.suggest_int('gamma', 2, 5)
        # 'augmentations': [
        #     ["mirror", trial.suggest_categorical('mirror_aug', [True, False])],
        #     ["shift", trial.suggest_categorical('shift_aug', [True, False])]]
    }
    combinations = generate_augmentation_options()
    # Let Optuna choose one combination
    aug_choice = trial.suggest_categorical('aug_choice', list(range(len(combinations))))
    chosen_combo = combinations[aug_choice]

    if chosen_combo is not None:
        params['augmentations'] = chosen_combo
    elif "augmentations" in params:
        del params["augmentations"]

    # Feature groups to consider
    emonet_features_options = [["embedding"], ["valence_arousal"], ["embedding", "valence_arousal"]]
    # choice = trial.suggest_categorical("emonet_features", [0, 1, 2])
    choice = 1
    feature_groups = emonet_features_options[choice] + [
        "OpenFace_pose",
        "OpenFace_gaze",
        "OpenFace_blink",
        "OpenFace_au_r"
    ]
    params['selected_features'] = feature_groups
    return params


def create_trial_config(base_config: Dict, trial_params: Dict, trial_dir: Path) -> Path:
    """Create a new config file for this trial."""
    trial_config = base_config.copy()
    trial_config.update(trial_params)

    config_path = trial_dir / "config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(trial_config, f)

    return config_path


def run_fold_training(trial_dir: Path, splits_dir: Path, paths_config: Path,
                      num_jobs: int) -> float:
    """Run training on all folds using GNU parallel."""
    # Create models and runs directories in trial folder
    trial_models_dir = trial_dir / "models"
    trial_runs_dir = trial_dir / "runs"
    trial_models_dir.mkdir(exist_ok=True)
    trial_runs_dir.mkdir(exist_ok=True)

    # Temporarily modify paths for this trial
    with open(paths_config) as f:
        paths_dict = json.load(f)

    # Create temporary paths config for this trial
    trial_paths_config = trial_dir / "paths.json"
    paths_dict["TrainedModels"] = str(trial_models_dir)
    paths_dict["RunsPath"] = str(trial_runs_dir)

    with open(trial_paths_config, 'w') as f:
        json.dump(paths_dict, f)

    # Create bash command using the trial-specific paths config
    cmd = f"""
    NUM_FOLDS=$(ls "{splits_dir}"/*.csv | wc -l)
    NUM_CORES=$(nproc)
    NUM_JOBS=$(( {num_jobs} < NUM_CORES ? {num_jobs} : NUM_CORES ))
    seq 0 $((NUM_FOLDS-1)) | parallel -j $NUM_JOBS python train_fold.py {{}} \
        --paths-config {trial_paths_config} \
        --training-config {trial_dir}/config.yml \
        --splits-dir {splits_dir}
    """

    # Run the command
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Warning: Some folds may have failed. stderr: {process.stderr}")

    paths_dict["configPath"] = trial_dir / "config.yml"
    paths_dict["splits_dir"] = splits_dir
    return paths_dict


def evaluate_ensemble(paths_dict):
    eval_object = EnsembleEvaluation(
        data_root=Path(paths_dict["FeaturesRoot"]),
        models_root=Path(paths_dict["TrainedModels"]),
        config_path=Path(paths_dict["configPath"]),
        splits_root=Path(paths_dict["splits_dir"]),
        eval_split="holdout"
    )
    ensemble_preds = eval_object.generate_ensemble_predictions()
    return f1_score(ensemble_preds.y_true_binary, (ensemble_preds.soft > 0.5).astype(int), average="macro")


def calculate_model_size_tier(params):
    tier = 0
    tier += params["batch_size"] == 128
    tier += (params["nheads"] == 4) / 2
    tier += params["project_to"] == 128
    tier += params["hidden_size"] == 128
    tier += (params["num_layers"] == 3) / 2
    return tier


def get_avg_validation_f1(trial_models_dir):
    trial_models_dir = Path(trial_models_dir)
    result_files = list(trial_models_dir.glob("*val.csv"))
    if not result_files:
        raise RuntimeError("No validation results found")

    # Load and aggregate results
    results_df = pd.concat([
        pd.read_csv(f) for f in trial_models_dir.glob("*val.csv")
    ])
    # Calculate mean macro F1 score
    macro_f1 = results_df[results_df['metric'] == 'f1-score']['macro avg'].mean()
    return macro_f1


def objective(trial: optuna.Trial, base_config: Dict, splits_dir: Path, paths_config: Path, output_dir: Path) -> \
        Optional[float]:
    """Optuna objective function."""
    # Create trial directory
    trial_dir = output_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Get hyperparameters for this trial
    trial_params = suggest_hyperparameters(trial)
    # Create config file for this trial
    create_trial_config(base_config, trial_params, trial_dir)

    # Run training
    try:
        model_size_tier = calculate_model_size_tier(trial_params)
        if model_size_tier < 1:
            num_jobs = 4
        elif model_size_tier <= 2:
            num_jobs = 3
        else:
            num_jobs = 2
        paths_dict = run_fold_training(trial_dir, splits_dir, paths_config, num_jobs=num_jobs)
        macro_f1 = get_avg_validation_f1(paths_dict["TrainedModels"])
        # macro_ensemble_f1 = evaluate_ensemble(paths_dict)
    except Exception as e:
        print(optuna.exceptions.TrialPruned(f"Trial failed due to: {type(e)}: {str(e)}"))
        return
    return macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--paths-config", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=42)
    args = parser.parse_args()

    # Load base configuration
    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Create study
    study = optuna.create_study(
        study_name="webcam_engagement_classifier",
        direction="maximize",
        storage=f"sqlite:///{args.output_dir}/optuna.db",
        load_if_exists=True
    )

    # Run optimization with progress tracking
    study.optimize(
        lambda trial: objective(
            trial,
            base_config,
            args.splits_dir,
            args.paths_config,
            args.output_dir
        ),
        n_trials=args.n_trials,
        show_progress_bar=True  # This shows a progress bar for each trial
    )

    # Print final results
    print("\nOptimization finished!")
    print(f"Best macro F1: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save best parameters
    with open(args.output_dir / "best_params.yml", 'w') as f:
        yaml.dump(study.best_params, f)


if __name__ == "__main__":
    """
    Example usage:
    
    python optimize_hyperparams.py \
    --base-config ../configs/base_config.yml \
    --paths-config ../configs/paths.json \
    --splits-dir ../configs/splits/fold_0 \
    --output-dir ../results/optimization/fold_0 \
    --n-trials 50
    """
    main()
