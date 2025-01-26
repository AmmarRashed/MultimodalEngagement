import os
import shutil
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloaders.webcam_features_dataset import WebcamFeaturesDataset, collate_fn
from models.attentive_sequential_classifier import AttentiveSequentialClassifier
from models.balanced_transformer import BalancedTransformerClassifier, FocalLoss, OrdinalBCELoss


def read_arguments_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        arguments = yaml.safe_load(file)
    return arguments


class EvaluationRun(object):
    def __init__(self, configs, estimator_name="Transformer", targets=['engagement'], exp_name=None,
                 output_path=None,
                 exp_tag='',
                 verbose=True
                 ):
        """
        Args:
            configs: dictionary
            estimator_name: name of estimator 'Transformer or AttentiveSequential
            targets: list of targets to predict
            exp_name: by Default <estimator_name>_<targets>
        """
        self.verbose = verbose
        self.log_line = dict()

        self.configs = configs
        self.estimator_name = estimator_name
        self.targets = targets
        if exp_name is None:
            exp_name = f"{estimator_name}_{'_'.join(targets)}"
        self.exp_name = exp_name
        self.model_name = f"{self.exp_name}.pth"
        if output_path is None:
            output_path = self.model_name
        self.output_path = output_path
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        self.data_splits = pd.read_csv(configs["splits_path"]).astype(str)
        exp_dir = os.path.join(configs["runs_path"], "/".join([self.exp_name, exp_tag]))
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
        self.writer = SummaryWriter(exp_dir)
        self.data_splits = pd.read_csv(configs["splits_path"]).astype(str)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_estimators = self.configs.get('n_estimators', 1)
        self.log_text('device', str(self.device))
        self.binary_targets = torch.tensor([1 if i > 2 else 0
                      for i in range(5)]).float().to(self.device)

    def run(self):
        self.load_data()
        self.define_model()
        self.training_setup()
        train_losses, validation_losses = self.training_loop()
        self.plot_losses(train_losses, validation_losses)
        self.evaluate(test_dataloader=self.validation_dataloader, suffix="_val")
        self.evaluate()

    def load_data(self):
        def init_dataset(split, config=None):
            if config is None:
                config = self.configs
            return self.load_dataset(configs=config,
                                     splits_df=self.data_splits,
                                     split=split,
                                     targets=self.targets)

        # Recalculating the labels mapper according to the labels in the training set
        self.train_dataset = init_dataset("train")
        self.validation_dataset = init_dataset("validation")
        self.test_dataset = init_dataset("test")

        self.class_weights = self.train_dataset.get_class_weights()

        aug = self.configs.get('augmentations', None)

        self.log_text("Class Weights", str(self.class_weights))

        mean, std = self.train_dataset.compute_features_mean_std()
        self.train_dataset.set_scaler(mean, std)
        self.validation_dataset.set_scaler(mean, std)
        self.test_dataset.set_scaler(mean, std)

        if aug is not None:
            self.train_dataset.augment_data(aug)

        n_train = len(self.train_dataset)
        n_validation = len(self.validation_dataset)
        n_test = len(self.test_dataset)
        total = n_train + n_validation + n_test

        self.log_text("DataSize/Train", f"{n_train} ({round(n_train / total * 100, 2)}%)")
        self.log_text("DataSize/Validation", f"{n_validation} ({round(n_validation / total * 100, 2)}%)")
        self.log_text("DataSize/Test", f"{n_test} ({round(n_test / total * 100, 2)}%)")

        batch_size = self.configs["batch_size"]
        # sampler=get_balanced_sampler(self.train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        for x, y, _ in self.test_dataloader:
            self.input_size = x.size(-1)
            self.num_targets = y.size(-1)
            break

        # self.output_size = self.train_dataset.num_classes  # Number of classes
        self.output_size = 1
        self.log_text("Input size", str(self.input_size))

    def log_text(self, tag, text):
        self.log_line.setdefault(tag, 0)
        self.writer.add_text(tag, str(text), self.log_line.get(tag))
        self.writer.flush()
        self.log_line[tag] += 1

    @staticmethod
    def load_dataset(configs, splits_df, split, targets=["engagement"]):
        kwargs = dict(root_dir=configs['root'],
                      labels_mapper=configs.get('labels_mapper'),
                      sampling_rate=configs['sampling_rate'],
                      max_frames=configs['max_frames'],
                      selected_features=configs['selected_features'],
                      targets=targets)
        if configs['split_by'] == "participant":
            kwargs["selected_participants"] = set(
                splits_df[splits_df.Split == split].ParticipantId.astype(str))
        else:
            kwargs["selected_sessions"] = set(
                splits_df[splits_df.Split == split].SessionId.astype(str))
        dataset = WebcamFeaturesDataset(**kwargs)
        dataset.set_mirrored_data_dir(configs.get('mirrored_root', None))
        return dataset

    def define_model(self):
        model_kwargs = dict(input_size=self.input_size,
                            hidden_size=self.configs['hidden_size'],
                            output_size=self.output_size,
                            num_targets=self.num_targets,
                            dropout=self.configs["dropout"],
                            num_layers=self.configs["num_layers"],
                            project_to=self.configs.get("project_to", None)
                            )
        if self.estimator_name == "Transformer":
            estimator = BalancedTransformerClassifier(nheads=self.configs.get("nheads", 1), **model_kwargs)
        elif self.estimator_name == "AttentiveSequential":
            estimator = AttentiveSequentialClassifier(**model_kwargs)
        else:
            raise Exception("Unknown estimator")

        if self.n_estimators == 1:
            self.model = estimator
        else:
            # TODO
            # self.model = SameModelVotingEnsemble()
            raise NotImplementedError

        self.model.to(self.device)

    def training_setup(self):
        if self.output_size == 1:
            loss_type = self.configs.get("loss", "bce")
            if loss_type == "focal":
                self.criterions = [
                    FocalLoss(
                        alpha=torch.tensor(cw[1] / cw[0], dtype=torch.float32).to(self.device) if self.configs.get(
                            "class_weights", True) else 1,
                        gamma=torch.tensor(self.configs.get("gamma", 2.), dtype=torch.float32).to(self.device))
                    for cw in self.class_weights]
            elif loss_type == "bce":
                self.criterions = [
                    BCEWithLogitsLoss(
                        pos_weight=torch.tensor(cw[1] / cw[0], dtype=torch.float32).to(self.device) if self.configs.get(
                            "class_weights", True) else None)
                    for cw in self.class_weights]
            elif loss_type == "bce_ordinal":
                self.criterions = [
                    OrdinalBCELoss(
                        pos_weight=torch.tensor(cw[1] / cw[0], dtype=torch.float32).to(self.device) if self.configs.get(
                            "class_weights", True) else None,
                        threshold=2,
                        alpha=torch.tensor(self.configs.get("alpha", 0.5))
                    )
                    for cw in self.class_weights]
        else:
            self.criterions = [nn.CrossEntropyLoss(
                weight=torch.tensor(cw, dtype=torch.float32).to(self.device) if self.configs.get("class_weights",
                                                                                                 True) else None)
                for cw in self.class_weights]
        # print(f"Pos Weight is: {self.class_weights[0][1] / self.class_weights[0][0]}")
        if self.n_estimators > 1:
            self.model.set_optimizer(optim.Adam, lr=self.configs["lr"], weight_decay=self.configs["decay"])
            self.model.set_criterion(self.criterions)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["lr"],
                                        weight_decay=self.configs["decay"])

    def ensemble_train_epoch(self):
        # TODO
        raise NotImplementedError

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels, lengths in (
                tqdm(self.train_dataloader, desc="Training", leave=False) if self.verbose else self.train_dataloader):
            x, y = inputs.to(self.device), labels.to(self.device)
            lengths = lengths.to(self.device)
            self.optimizer.zero_grad()
            # Forward pass
            loss = self.forward(x, y, lengths=lengths)
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            # clipping gradients
            # nn.utils.clip_grad_norm_(model.parameters(), 6.0)
            # optimization
            self.optimizer.step()
        return total_loss / len(self.train_dataloader)

    def validate_ensemble(self):
        # TODO
        raise NotImplementedError

    def forward(self, x, y, lengths=None):
        output = self.model(x, lengths)
        loss = 0
        for i, yhat in enumerate(output):
            index = y[:, i] != -1
            y_true = y[index, i]
            y_pred = yhat[index]
            if self.output_size == 1:
                y_pred = y_pred.view(-1)
                y_true = y_true.to(torch.float32)
            loss += self.criterions[i](y_pred, y_true)
        loss /= len(output)
        return loss

    def validate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.validation_dataloader
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels, lengths in (
                    tqdm(dataloader, desc="Validation", leave=False) if self.verbose else dataloader):
                x, y = inputs.to(self.device), labels.to(self.device)
                lengths = lengths.to(self.device)
                # Forward pass
                loss = self.forward(x, y, lengths=lengths)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    # TODO
    def ensemble_training_loop(self):
        raise NotImplementedError

    def training_loop(self):
        num_epochs = self.configs.get('epochs', 100)
        best_validation_loss = self.validate()
        max_patience = self.configs.get('patience', 10)
        patience = max_patience
        best_epoch = -1
        torch.save(self.model.state_dict(), self.output_path + '.pth')
        losses = {best_epoch: (self.validate(self.train_dataloader), best_validation_loss)}

        train_losses = list()
        validation_losses = list()
        for epoch in (tqdm(range(1, num_epochs + 1), desc="Epoch") if self.verbose else range(1, num_epochs + 1)):
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            validation_loss = self.validate()
            self.writer.add_scalars(f"Loss", {"train": train_loss, "validation": validation_loss}, epoch)
            validation_losses.append(validation_loss)
            self.writer.flush()
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                losses[epoch] = (train_loss, validation_loss)
                patience = max_patience
                torch.save(self.model.state_dict(), self.output_path + '.pth')
            else:
                patience -= 1
            if patience == 0:
                break
        train_loss, validation_loss = losses[best_epoch]
        self.log_text('BestEpoch/Epoch', best_epoch)
        self.log_text('BestEpoch/TrainLoss', train_loss)
        self.log_text('BestEpoch/ValidationLoss', validation_loss)

        return train_losses, validation_losses

    def plot_losses(self, train_losses, validation_losses):
        fig = plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(validation_losses, label="Validation")
        plt.legend()
        self.writer.add_figure("Loss", fig, 0)

    def evaluate(self, test_dataloader=None, suffix=""):
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        state_dict = torch.load(self.output_path + '.pth')
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        result = {target: {'y_pred': [], 'y_true': []} for target in self.targets}
        with torch.no_grad():
            for inputs, labels, lengths in (
                    tqdm(test_dataloader, desc="Test", leave=False) if self.verbose else test_dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                outputs = self.model(inputs, lengths=lengths)
                for i, (target, yhat) in enumerate(zip(self.targets, outputs)):
                    y_true = labels.cpu().numpy()[:, i]
                    index = y_true != -1
                    if self.output_size == 1:
                        y_pred_prob = torch.sigmoid(yhat[index])
                        y_pred = (y_pred_prob > 0.5).cpu().numpy().astype(int)
                    else:
                        y_pred = yhat[index].argmax(dim=1).cpu().numpy().astype(int)
                    y_true = y_true[index]

                    # binarize labels
                    if self.configs.get("labels_mapper") is None:
                        y_true = self.binary_targets[y_true].cpu().numpy().astype(int)

                    result[target]['y_pred'].extend(y_pred)
                    result[target]['y_true'].extend(y_true)

        for t in self.targets:
            y_true = result[t]["y_true"]
            y_pred = result[t]["y_pred"]
            clf_report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).fillna(0)
            result[t]["report"] = clf_report
            clf_report.to_csv(f"{self.output_path}_{t}{suffix}.csv", index=True, index_label="metric")
            self.log_text(f'clf_report/{t}', clf_report.to_markdown())


if __name__ == "__main__":
    configs = read_arguments_from_yaml("common_configs.yml")
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', default='Transformer')
    parser.add_argument('-n', '--targets', nargs='+', default=['engagement'])

    args = parser.parse_args()

    experiment = EvaluationRun(configs, estimator_name=args.model, targets=args.targets)
    print(f"Running experiment: {experiment.exp_name}")
    experiment.run()
