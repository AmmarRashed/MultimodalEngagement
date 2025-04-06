from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import DataLoader

from dataloaders.multimodal_dataloader import *
from models.multimodal_fusion import *


class MultimodalExperiment:
    DEFAULT_MODALITIES = ["EEG", "EYE", "OpenFace"]

    def __init__(self, dataset_root, outer_fold: str, device=None, model_kwargs={"dropout": 0.3, "hidden_dim": 128},
                 modalities=None,
                 inner_fold: int = 0
                 ):
        """

        :param dataset_root:
        :param outer_fold:
        :param device:
        :param model_kwargs:
        :param modalities: {modality: (drop column flag, list of columns)}
        :param inner_fold:
        """
        if modalities is None:
            modalities = self.DEFAULT_MODALITIES
        self.dataset_root = Path(dataset_root)
        self.splits_path = self.dataset_root / "splits" / outer_fold / f"{inner_fold}.csv"
        splits_df = pd.read_csv(self.splits_path)
        self.splits_df = splits_df
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.modalities = modalities
        self.train_dataset, self.train_dataloader = self.init_dataloader("train", modalities=modalities)
        self.val_dataset, self.val_dataloader = self.init_dataloader("validation", scalers=self.train_dataset.scalers,
                                                                     fit_scaler=False, shuffle=False,
                                                                     modalities=modalities
                                                                     )
        self.test_dataset, self.test_dataloader = self.init_dataloader("test", scalers=self.train_dataset.scalers,
                                                                       fit_scaler=False, shuffle=False,
                                                                       modalities=modalities
                                                                       )
        self.pos_weight = torch.tensor(
            sum(self.train_dataset.y_transformed == 0) / sum(self.train_dataset.y_transformed == 1)).squeeze()

        self.model = self.init_model(**model_kwargs)

    def init_dataloader(self, split, scalers=None, fit_scaler=True, shuffle=True, modalities=None):
        """

        :param split:
        :param scalers:
        :param fit_scaler:
        :param shuffle:
        :param modalities: {modality: (drop column flag, list of columns)}
        :return:
        """
        if modalities is None:
            modalities = self.DEFAULT_MODALITIES
        participants = self.splits_df[self.splits_df.Split == split].ParticipantId.unique()
        dataset = MultiModalDataset(
            self.dataset_root / "Samples",
            modalities=modalities,
            participants=participants,
            targets=["engagement"],
            scalers=scalers,
            fit_scaler=fit_scaler,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            collate_fn=MultimodalCollator(),
            shuffle=shuffle,
            pin_memory=True,
        )
        return dataset, dataloader

    def init_model(self, **kwargs):
        input_dims = [self.train_dataset.X[i][0].shape[-1] for i in range(len(self.train_dataset.X))]
        model = MultimodalFusion(input_dims, **kwargs)
        model.to(self.device)
        return model

    def run_training(self, **training_kwargs):
        model, history = train_model(self.model, self.train_dataloader, self.val_dataloader, self.device,
                                     pos_weight=self.pos_weight,
                                     **training_kwargs)
        self.model = model
        return history

    def evaluate_model(self):
        y_true, y_pred, y_pred_prob = get_predictions(self.model, self.test_dataloader, self.device)
        report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
        report = report.assign(roc_auc=roc_auc_score(y_true, y_pred_prob))
        return report.rename(columns={"0": "Low", "1": "High"})
