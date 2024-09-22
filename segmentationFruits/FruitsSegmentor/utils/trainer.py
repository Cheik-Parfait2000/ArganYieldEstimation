from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from segmentation_models_pytorch.losses import JaccardLoss
import pandas as pd

from FruitsSegmentor.utils.dataset_dataloader_utilities import SegmentationDataset


class Trainer(object):
    def __init__(self, _data, model, epochs, save_dir, loss=None, optimizer=None, learning_rate=0.005):
        self.data = _data
        self.model = model
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.early_stopper = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) if optimizer is None else optimizer
        self.learning_rate = learning_rate

        mode = "multiclass" if self.model.model_building_config["classes"] >= 3 else "binary"
        self.loss = JaccardLoss(mode=mode, from_logits=True, eps=1e-6, log_loss=True) if loss is None else loss

    def compute_loss(self, logits, masks):
        """
        Compute the Dice loss for the given logits and masks

        logits: tensor of shape (batch_size, num_classes, height, width)
        masks: tensor of shape (batch_size, height, width)
        mode: "multiclass" or "binary"
        """

        loss = self.loss(logits, masks)
        return loss

    def train_epoch(self, data_loader, optimizer, epoch, n_epochs):
        """
        Train the model for one epoch
        """
        self.model.train()
        self.model.to(self.device)
        total_loss = 0
        loop_train = tqdm(data_loader, desc=f"Epoch {epoch} / {n_epochs}")
        for images, masks in loop_train:
            images, masks = images.to(self.device), masks.to(self.device)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.compute_loss(logits, masks)
            loss.backward()  # Calculer les gradients de la fonction coût par rapport au paramètres du modèle
            optimizer.step()  # Mettre à jour les paramètres du modèle

            total_loss += (loss.item() / len(data_loader))
            loss_format = "{:.5f}".format(total_loss)
            loop_train.set_description(f"Epoch {epoch + 1} / {n_epochs}     Training... Loss = {loss_format} ")
        return total_loss

    def val_epoch(self, data_loader):
        """
        Evaluate the model for one epoch
        """
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            loop_eval = tqdm(data_loader, desc="evaluating")
            for images, masks in loop_eval:
                images, masks = images.to(self.device), masks.to(self.device)
                logits = self.model(images)
                loss = self.compute_loss(logits, masks)

                total_loss += (loss.item() / len(data_loader))
                loss_format = "{:.5f}".format(total_loss)
                loop_eval.set_description(f"               Validation... loss = {loss_format} ")
            return total_loss

    def train(self, train_checkpoint=None, patience=50):
        """
          Train the model for all the epochs
        """
        # Création des dossiers pour la sauvegarde des résultats d'entrainement
        Path(self.save_dir).mkdir(exist_ok=True)
        (Path(self.save_dir) / "weights").mkdir(exist_ok=True)
        weights_path = str(Path(self.save_dir) / "weights").replace("\\", "/")
        save_dir = str(Path(self.save_dir)).replace("\\", "/")
        # ----------------------- Get the train and val dataloaders ---------------
        train_loader = SegmentationDataset(self.data["train"]).load_data()
        val_loader = SegmentationDataset(self.data["val"]).load_data()

        # data_loader, optimizer, epoch, n_epochs

        LR = self.learning_rate
        optimizer = self.optimizer
        best_val_loss = 0
        start = 0
        end = self.epochs
        df = pd.DataFrame({
            "epoch": [],
            "train_loss": [],
            "val_loss": []
        })
        """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
             optimizer, mode="min", factor=0.5, threshold=0.001, patience=10
        )"""

        cols = ["epoch", "train_loss", "val_loss"]
        if train_checkpoint is not None:
            ckpt = torch.load(train_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            start = ckpt["epoch"] + 1
            end = start + self.epochs + 1
            best_val_loss = ckpt["best_val_loss"]
            df = pd.read_csv(save_dir + "/results.csv")
            df = df[cols]

        # ----------- initialize the early stopper ------------
        # self.early_stopper = EarlyStopping(patience=patience)
        # ---------------------- Start traning ------------------------------------
        for epoch in range(start, end):

            best_val_loss = 0
            train_loss = self.train_epoch(train_loader, optimizer, epoch, self.epochs)
            val_loss = self.val_epoch(val_loader)

            # scheduler.step(val_loss)

            if epoch == 0:
                best_val_loss = val_loss
            elif val_loss < best_val_loss:
                best_val_loss = val_loss

            df.loc[len(df.index)] = [epoch, train_loss, val_loss]
            df.to_csv(save_dir + "/results.csv")
            self.model.save(weights_path + "/best.pt")
            self.model.save(weights_path + "/last.pt")
            self.model.save(weights_path + "/train_checkpoint.pt",
                            {
                                "epoch": epoch,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_val_loss": best_val_loss,
                                "model_state_dict": self.model.state_dict()
                            })
            print()

        # Message de fin d'entrainement
        print(f"Fin d'entrainement du modèle. Résultats sauvegardés à {self.save_dir}")


def compute_metrics(stats, dict_metrics={"accuracy": smp.metrics.functional.accuracy, \
                                         "iou_score": smp.metrics.functional.iou_score,
                                         "f1_score": smp.metrics.functional.f1_score}, \
                    classes_names=["global", "background", "fruit", "edge"], classes_indexes=[-1, 0, 1, 2]):
    """
    Compute some metrics for all the classes and the overall score

    return a dict of metrics:
      {metric_name_class_name: metric_value}
    """
    metrics_keys = dict_metrics.keys()
    metrics_values = {f"{metrics_key}_{class_name}": None for metrics_key in metrics_keys for class_name in
                      classes_names}

    for metric_key in metrics_keys:
        metric_function = dict_metrics[metric_key]
        for i, class_name in zip(classes_indexes, classes_names):
            if class_name == "global":
                global_metric = metric_function(stats[0], stats[1], stats[2], stats[3], reduction="micro")
                metrics_values[f"{metric_key}_{class_name}"] = global_metric.item()
            else:
                by_class_metric = metric_function(stats[0].sum(dim=0), stats[1].sum(dim=0), stats[2].sum(dim=0),
                                                  stats[3].sum(dim=0), reduction=None)
                metrics_values[f"{metric_key}_{class_name}"] = by_class_metric[i].item()

    return metrics_values

