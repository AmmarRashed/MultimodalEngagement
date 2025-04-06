import json
import os
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from IPython import display

from experiments.multimodal_experiment import MultimodalExperiment


class Paths:
    """Utility class for accessing paths"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.paths = json.load(f)

    def __getitem__(self, key):
        """Allow dictionary-style access to paths"""
        return Path(self.paths[key])

    def __getattr__(self, name):
        """Allow attribute-style access to paths"""
        try:
            return Path(self.paths[name])
        except KeyError:
            raise AttributeError(f"No path found for '{name}'")


def combine_avg_std_tables_latex(avg_df, std_df, stack=False, low_col="0", high_col="1"):
    print(f"Accuracy: {avg_df['accuracy'].iloc[0]} ± {std_df['accuracy'].iloc[0]}")
    if "roc_auc" in avg_df:
        print(f"ROC-AUC: {avg_df['roc_auc'].iloc[0]} ± {std_df['roc_auc'].iloc[0]}")
    new_df = dict()
    for k in avg_df:
        new_df[k] = avg_df[k].round(2).astype(str) + u" \u00B1 " + std_df[k].round(2).astype(str)
    new_df = pd.DataFrame(new_df).loc[["precision", "recall", "f1-score", "support"]]
    new_df = new_df[[low_col, high_col, "macro avg"]].rename(
        columns={low_col: "Low", high_col: "High", "macro avg": "Avg."})
    if stack:
        return new_df.transpose().stack().to_frame().transpose()
    return new_df


def show_results(root, combine=True, stack=False, suffix=""):
    """
    Go over result tables in an experiments results root, and displays them in a jupyter notebook
    """
    results = pd.concat([
        pd.read_csv(os.path.join(root, f))
        for f in os.listdir(root) if f.endswith(f'{suffix}.csv')
    ])
    # Average Results
    mean = results.groupby("metric").mean().loc[["precision", "recall", "f1-score", "support"], :]

    # Standard Deviation
    std = results.groupby("metric").std().loc[["precision", "recall", "f1-score", "support"], :]
    if combine:
        return combine_avg_std_tables_latex(mean, std, stack)
    print("Mean")
    display(mean)
    print("STD")
    display(std)
    return mean, std


def read_yaml_config(config_path: Path) -> Dict:
    """Read and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict containing the configuration

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise


def read_reports(root):
    reports = []
    for file in Path(root).glob("*.csv"):
        reports.append(pd.read_csv(file))
    reports_df = pd.concat(reports)
    avg_report = reports_df.groupby("Metric").mean()
    std_report = reports_df.groupby("Metric").std()
    final_report = combine_avg_std_tables_latex(avg_report, std_report, low_col="Low", high_col="High")
    return final_report


def run_experiment(dataset_root, modalities=None, results_dir=Path("../results"), exp_kwargs=dict(), **training_kwargs):
    if modalities is None:
        modalities = MultimodalExperiment.DEFAULT_MODALITIES
    splits_root = Path(dataset_root) / "splits"
    exp_prefix = '_'.join([f.lower() for f in modalities])
    tables_dir = Path(results_dir / "tables" / f"{exp_prefix}/")
    os.makedirs(tables_dir, exist_ok=True)
    history_dir = Path(results_dir / "history" / f"{exp_prefix}/")
    os.makedirs(history_dir, exist_ok=True)
    reports = []
    for i, folder in enumerate(sorted(splits_root.glob("*/"))):
        print(f"Running Experiment on Fold {i}")
        exp = MultimodalExperiment(dataset_root=dataset_root, outer_fold=folder.stem, inner_fold=0,
                                   modalities=modalities,
                                   **exp_kwargs)
        history = exp.run_training(best_model_path=f"../results/models/{exp_prefix}/{i}.pth", **training_kwargs)
        pickle.dump(history, open(history_dir / f"{i}.pkl", 'wb'))
        report = exp.evaluate_model().reset_index().rename(columns={"index": "Metric"})
        report.to_csv(tables_dir / f"{i}.csv", index=None)
        reports.append(report)
    return read_reports(tables_dir)
