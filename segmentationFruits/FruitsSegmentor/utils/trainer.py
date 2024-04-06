from pathlib import Path

import torch
from segmentation_models_pytorch.losses import DiceLoss
import segmentation_models_pytorch as smp
from tqdm import tqdm

from FruitsSegmentor.utils.dataset_dataloader_utilities import SegmentationDataset


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


def compute_loss(logits, masks, mode="multiclass"):
    """
  Compute the Dice loss for the given logits and masks

  logits: tensor of shape (batch_size, num_classes, height, width)
  masks: tensor of shape (batch_size, height, width)
  mode: "multiclass" or "binary"
  """
    loss = DiceLoss(mode="multiclass", from_logits=True)(logits, masks)
    loss_format = "{:.2f}".format(loss)
    return loss, loss_format


class Trainer(object):
    def __init__(self, _data, model):
        self.data = _data
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_loss(self, logits, masks):
        """
        Compute the Dice loss for the given logits and masks

        logits: tensor of shape (batch_size, num_classes, height, width)
        masks: tensor of shape (batch_size, height, width)
        mode: "multiclass" or "binary"
        """
        mode = "multiclass" if self.model.model_building_config["classes"] == 3 else "binary"
        loss = DiceLoss(mode=mode, from_logits=True)(logits, masks)
        return loss

    def train_epoch(self, data_loader, optimizer, epoch, n_epochs):
        """
        Train the model for one epoch
        """
        self.model.train()
        total_loss = 0
        loop_train = tqdm(data_loader, desc="training")
        for images, masks in loop_train:
            images, masks = images.to(self.device), masks.to(self.device)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.compute_loss(logits, masks)
            loss.backward()  # Calculer les gradients de la fonction coût par rapport au paramètres du modèle
            optimizer.step()  # Mettre à jour les paramètres du modèle

            total_loss += (loss.item() / len(data_loader))
            loss_format = "{:.2f}".format(total_loss)
            loop_train.set_description(f"Epoch {epoch+1} / {n_epochs}     Training... dice_loss = {loss_format}")
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
                loss_format = "{:.2f}".format(total_loss)
                loop_eval.set_description(f"               Validation... dice_loss = {loss_format} ")
            return total_loss

    def train(self, learning_rate=0.005, epochs=100, save_dir="", optimizer=None, lr_scheduler=None,
              resume_training=True):
        """
            Train the model for all the epochs
        """
        (Path(save_dir) / "weights").mkdir(exist_ok=True)
        weights_path = str(Path(save_dir) / "weights").replace("\\", "/")
        save_dir = str(Path(save_dir)).replace("\\", "/")

        # ----------------------- Get the train and val dataloaders ---------------
        train_loader = SegmentationDataset(self.data["train"]).load_data()
        val_loader = SegmentationDataset(self.data["val"]).load_data()

        LR = learning_rate
        if optimizer is not None:
            optimizer = optimizer(self.model.parameters(), lr=LR)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        if lr_scheduler is not None:
            scheduler = lr_scheduler(optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

        # ---------------------- Start traning ------------------------------------
        with open(save_dir + "/results.csv", 'w') as f:
            f.write("epoch,train_loss,val_loss\n")
            best_val_loss = 0
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader, optimizer, epoch, epochs)
                val_loss = self.val_epoch(val_loader)

                # Call the scheduler to monitor to optimizer learning_rate
                scheduler.step(val_loss)

                f.write(f"{epoch+1},{train_loss},{val_loss}\n")
                if epoch == 0:
                    best_val_loss = val_loss
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save(weights_path + "/best.pt")

                self.model.save(weights_path + "/last.pt")
                # Save the model with the current epoch and the optimizer state
                if resume_training:
                    self.model.save(weights_path + "/train_checkpoint.pt",
                                    {
                                        "epoch": epoch,
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "best_val_loss": best_val_loss,
                                    })
