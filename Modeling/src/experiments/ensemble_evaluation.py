from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from dataloaders.webcam_features_dataset import collate_fn
from experiments.evaluation_run import EvaluationRun
from models.balanced_transformer import BalancedTransformerClassifier
from utils import read_yaml_config


class EnsembleEvaluation:
    def __init__(self, data_root: Path, models_root: Path, config_path: Path, splits_root: Path,
                 reset_label_mapper=True,
                 verbose=False,
                 eval_split="test"
                 ):
        self.data_root = Path(data_root)
        self.models_root = Path(models_root)
        self.configs = read_yaml_config(config_path)
        self.configs["root"] = self.data_root
        self.splits_root = Path(splits_root)
        self.reset_label_mapper = reset_label_mapper
        if self.reset_label_mapper:
            self.configs["labels_mapper"] = None
        self.verbose = verbose
        self.eval_split = eval_split

        self.inner_folds = sorted([fold.stem for fold in self.splits_root.glob("*.csv")])

        self.eval_dataset = self.load_eval_dataset()

        x, y = self.eval_dataset[0]
        self.input_size = x.size(-1)
        self.num_targets = y.size(-1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_eval_dataset(self):
        splits_df = pd.read_csv(self.splits_root / f"{self.inner_folds[-1]}.csv").astype(str)
        eval_dataset = EvaluationRun.load_dataset(
            configs=self.configs,
            splits_df=splits_df,
            split=self.eval_split)
        return eval_dataset

    def get_folds_scaler_weights(self, fold):
        data_splits = pd.read_csv(self.splits_root / f"{fold}.csv").astype(str)
        train_dataset = EvaluationRun.load_dataset(
            configs=self.configs,
            splits_df=data_splits,
            split="train")
        # apply scaling
        mean, std = train_dataset.compute_features_mean_std()
        return mean, std

    def get_scaled_eval_dataloader(self, fold):
        mean, std = self.get_folds_scaler_weights(fold)
        eval_dataset = deepcopy(self.eval_dataset)
        eval_dataset.set_scaler(mean, std)
        # Shuffle=False --> Because, we want to align predictions across inner folds
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.configs['batch_size'],
                                     shuffle=False,  # Important to put as False.
                                     collate_fn=collate_fn)
        return eval_dataloader

    def init_model_image(self, fold):
        model_kwargs = dict(input_size=self.input_size,
                            hidden_size=self.configs['hidden_size'],
                            output_size=1,
                            num_targets=self.num_targets,
                            dropout=self.configs["dropout"],
                            num_layers=self.configs["num_layers"],
                            project_to=self.configs.get("project_to", None))
        model = BalancedTransformerClassifier(nheads=self.configs.get("nheads", 1), **model_kwargs)
        try:
            state_dict_path = list(self.models_root.glob(f"*{fold}.pth"))[0]
        except IndexError:
            raise Exception(f"No model image found for fold {fold}")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        return model

    def load_fold(self, fold):
        """load the test dataloader and model image for an inner fold"""
        eval_dataloader = self.get_scaled_eval_dataloader(fold)
        model = self.init_model_image(fold)
        return model, eval_dataloader

    def _generate_predictions(self, model, dataloader):
        """Run the model on the given dataloader and return prediction probabilities"""
        result = {target: {'y_pred': [], 'y_true': []} for target in self.eval_dataset.targets}
        model.eval()
        with torch.no_grad():
            for inputs, labels, lengths in (dataloader if self.verbose else dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                outputs = model(inputs, lengths=lengths)
                for i, (target, yhat) in enumerate(zip(self.eval_dataset.targets, outputs)):
                    y_true = labels.cpu().numpy()[:, i]
                    y_pred_prob = torch.sigmoid(yhat)[:, i].detach().cpu().tolist()
                    y_true = list(y_true)

                    result[target]['y_pred'].extend(y_pred_prob)
                    result[target]['y_true'].extend(y_true)
        return result

    def generate_fold_predictions(self, fold, target="engagement"):
        model, dataloader = self.load_fold(fold)
        result = self._generate_predictions(model, dataloader)
        return pd.DataFrame(result[target])

    def get_fold_validation_metric(self, fold, metric="f1-score"):
        df = pd.read_csv(self.models_root / f"TransformerEngagement_{fold}_engagement_val.csv")
        return df[df.metric == metric]["macro avg"]

    def get_all_folds_validation_metric(self, metric="f1-score"):
        df = pd.DataFrame()
        for fold in self.inner_folds:
            df = df.assign(**{f"{fold}_val_{metric}": self.get_fold_validation_metric(fold)})
        return df

    def generate_all_folds_predictions(self, skip_last=True, n_jobs=-1):
        folds = self.inner_folds[:-1] if skip_last else self.inner_folds

        def step(fold):
            df = self.generate_fold_predictions(fold)
            return pd.DataFrame({"y_true": df.y_true.astype(int), f"{fold}_pred": df.y_pred})

        results = Parallel(n_jobs=n_jobs)(delayed(step)(fold) for fold in folds)
        results = pd.concat((r.drop("y_true", axis=1) if i > 0 else r for i, r in enumerate(results)), axis=1)
        return results

    def generate_ensemble_predictions(self, calculate_weighted=False):
        predictions = self.generate_all_folds_predictions()
        columns = ["y_true", "y_true_binary", "soft", "hard"]

        predictions["soft"] = predictions[[f"{fold}_pred" for fold in range(6)]].mean(axis=1)
        predictions["hard"] = \
            pd.concat([predictions[f"{fold}_pred"].apply(lambda x: int(x > 0.5)) for fold in range(6)], axis=1).mode(
                axis=1)[0]

        if calculate_weighted:
            columns.append("weighted")
            weights = self.get_all_folds_validation_metric()
            for col in weights:
                predictions[col] = weights[col].iloc[0]
            predictions["weighted"] = pd.concat(
                [predictions[f"{fold}_pred"] * predictions[f"{fold}_val_f1-score"] for fold in range(6)], axis=1).sum(
                axis=1) / pd.concat([predictions[f"{fold}_val_f1-score"] for fold in range(6)], axis=1).sum(axis=1)

        if self.reset_label_mapper:
            predictions["y_true_binary"] = (predictions["y_true"] > 2).astype(int)
        return predictions[columns]
