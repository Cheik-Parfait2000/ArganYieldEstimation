# ================ Import modules ====================
import glob
import os

import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
import tqdm
from PIL import Image
from skimage.transform import resize

from pathlib import Path

from FruitsSegmentor.utils.dataset_dataloader_utilities import DatasetConfig
from FruitsSegmentor.utils.errors import InvalidFileError, InvalidShapeError
from FruitsSegmentor.utils.ops import get_device, preprocess_input_for_prediction, check_dict_keys_for_train, \
    find_closest_dividor, remove_smallest_blobs, post_process_mask
from FruitsSegmentor.utils.trainer import Trainer

from typing import List, Union, Tuple
import numpy as np
from skimage import io
import albumentations as A

DEFAULT_CONFIG = Path(__file__).parent.parent.resolve() / "configs" / "unet_config.yaml"
DEFAULT_CONFIG = str(DEFAULT_CONFIG).replace("\\", "/")


# ================= Build the main class ====================

class SegmentationModel(nn.Module):
    """Class de base pour la segmentation d'images"""

    def __init__(self, model: str = None, model_name="unet", **kwargs):
        """
    args:
      model :
        - pretrained_model : .pt file
      model_name : nom du l'architecture du modèle
      kwargs : arguments supplémentaires à mots clés à passer à l'architecture
        pour la création du modèle. Ces mots clés doivent être définis selon
        les arguments acceptés par le modèle de segmentation de
          segmentation_models_pytorch
    """
        super().__init__()
        self.model_config = model
        self.model_name = model_name
        self.model = None
        self.model_kwargs = kwargs
        if self.model_config:
            self.model_kwargs = dict()
        self.model_building_config = dict()
        self.trainer = None

        # Build the model form the configuration
        self._build_model()

    def __call__(self, x):
        """
    Effectuer une prédiction
    """
        return self.model(x)

    def forward(self, x):
        """
    Effectuer une prédiction
    """
        return self.model(x)

    def _build_model(self):
        """Build the model from a kwargs or from pretrained weights"""
        if self.model_config is None:
            self._build_model_from_kwargs()
        else:
            if Path(self.model_config).exists():
                if Path(self.model_config).suffix in (".pt", ".tar"):
                    self._build_pretrained_model(device="cpu")
                else:
                    raise InvalidFileError(f"{self.model} is not a recognized file.\n Accepted formats are .pt/.tar")
            else:
                raise FileNotFoundError(f"{self.model} was not found! please check the good path of your file")

    def _build_model_from_kwargs(self):
        """
        Build the model from kwargs
        """
        if self.model_name == "unet":
            self.model = smp.Unet(**self.model_kwargs)
            # Add a key to specify the models name :
            self.model_building_config["model_name"] = "unet"

            # Mettre à jours les paramètres de construction du modèle
            for key in self.model_kwargs:
                self.model_building_config[key] = self.model_kwargs[key]
        else:
            raise NotImplementedError("Model architecture not implemented!")

    def _restore_model_weights(self, weights, device="cpu"):
        """Restore a model from weights

     weights : path to the weights file or a dict containing the model weights
    """
        try:
            if isinstance(weights, dict):
                self.load_state_dict(weights)
            else:
                self.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        except:
            raise RuntimeWarning("Failed to load the model weights. Please make sure that the weights file is valid.")

    def _build_pretrained_model(self, device=get_device()):
        """
        Build the model from pretrained weights

        If the model config ends with .pt or .tar, the model will be built

        This file should contain these keys :
          "building_config" : the model configuration. A dict containing the keys
                        "model_name" along with the model architecture keys
          "model_state_dict" : the model weights

        """
        loaded_config = torch.load(self.model_config, map_location=device)
        self.model_building_config = loaded_config["building_config"]
        model_name = self.model_building_config["model_name"]
        arch_config = {
            key: self.model_building_config[key] for key in self.model_building_config
            if key != "model_name"
        }

        for key in arch_config:
            self.model_kwargs[key] = arch_config[key]

        if model_name == "unet":
            self._build_model_from_kwargs()
            model_weights = loaded_config["model_state_dict"]
            # Load the model state
            self._restore_model_weights(model_weights, device)

    def __repr__(self):
        return super().__repr__()

    def fit(self, donnees_dict: dict, classes_mapping: dict, augmentations: A.Compose = None,
            batch_size=8, image_size=640, save_dir=None, epochs=100,
            train_checkpoint=None, loss=None, **kwargs):
        """Train the model
    donnees_dict : the data to use for training :
      - train_images: images and masks for training
      - train_masks: masks for training
      - val_images: images and masks for validation
      - val_masks: masks for validation
    classes_mapping : a dict containing the classes mapping. Pixels values in the
    annotation images are mapped to the classes names. e.g. {0: 0, 255: 1}
    batch_size : the batch size to use for training
    image_size : the image size to use for training
    save_dir : the directory to save the model
    loss: loss function to use for training
    kwargs : arguments supplémentaires à mots clés à passer au Trainer pour l'entrainement.
             Les arguments acceptés sont :
             - optimizer : optimizer to use for training
             - learning_rate : learning rate to use for training
    """

        # vérifier la validité des paramètres
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        if not isinstance(donnees_dict, dict):
            raise TypeError("data must be a dictionary")
        else:
            check_dict_keys_for_train(dict_=donnees_dict,
                                      keys=["train_images", "train_masks", "val_images", "val_masks"])
        """if augmentations is not None:
      if not isinstance(augmentations, A.Compose):
        raise TypeError("augmentations must be an instance of A.Compose")"""
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if not isinstance(image_size, int):
            raise TypeError("image_size must be an integer")

        if not isinstance(classes_mapping, dict):
            raise TypeError("classes_mapping must be a dictionary")

        if save_dir is not None and not isinstance(save_dir, str):
            raise TypeError("save_dir must be a string")

        if not Path(save_dir).parent.exists():
            raise FileNotFoundError(f"{save_dir} not found! Check the directory!")
        else:
            Path(save_dir).mkdir(exist_ok=True)

        if save_dir is None:
            save_dir = str(Path().cwd() / "runs")
            Path(save_dir).mkdir(exist_ok=True)

        # --------------- Get train and val data configs --------------------
        n_classes = self.model_building_config["classes"]
        classes_pixels_values = list(classes_mapping.keys())
        if n_classes == 1:
            if len(classes_pixels_values) != 2:
                raise ValueError(
                    "The number of classes the model should output must be equal to the number of classes specified in the classes mapping!")
        else:
            if len(classes_pixels_values) != n_classes:
                raise ValueError(
                    "The number of classes the model should output must be equal to the number of classes specified in the classes mapping!")

        train_data_dict = {"images": donnees_dict["train_images"], "labels_masks": donnees_dict["train_masks"]}
        val_data_dict = {"images": donnees_dict["val_images"], "labels_masks": donnees_dict["val_masks"]}

        train_data_cfg = DatasetConfig(batch_size, image_size, augmentations, n_classes,
                                       classes_pixels_values, classes_mapping,
                                       data_config=train_data_dict)
        val_data_cfg = DatasetConfig(batch_size, image_size, None, n_classes,
                                     classes_pixels_values, classes_mapping,
                                     data_config=val_data_dict)
        # ----------------- Send the task to to trainer --------------- data, model, epochs, save_dir
        self.trainer = Trainer(model=self, _data={"train": train_data_cfg, "val": val_data_cfg},
                               epochs=epochs, save_dir=save_dir, loss=loss, **kwargs)
        self.trainer.train(train_checkpoint)

    def predict(self, image: Union[str, np.ndarray], image_size: Union[int, List, Tuple] = None, post_process=True, device="cpu"):
        """
        Effectuer une prédiction en utilisant le modèle pré-entrainé.
        image : chemin vers l'image à utilisé ou une image numpy
        prediction_size: taille à laquelle l'image doit être redimensionnée lors pour la prédiction
          - int : redimensions w et h à cette taille
          - list : [w, h]
          - tuple : (w, h)
        """
        if isinstance(image, str):
            if Path(image).exists():
                image_array = io.imread(image)
            else:
                raise FileNotFoundError(f"{image} was not found! please check the good path of your file")
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            raise TypeError("image must be a string or a numpy array")

        if image_size is None:
            image_size = [image_array.shape[1], image_array.shape[0]]

        if type(image_size) not in [int, list, tuple]:
            raise TypeError(
                "prediction_size must be either an integer or a list of 2 elements or a tuple of two elements")

        if not isinstance(image_size, int):
            if len(image_size) != 2:
                raise ValueError("prediction_size contain two elements : width and height")
            else:
                for e in image_size:
                    if not isinstance(e, int):
                        raise TypeError("Sequence values must be integers")
        else:
            image_size = [image_size, image_size]

        # Check the dimensions of the image : HWC with C=3
        if len(image_array.shape) != 3:
            raise InvalidShapeError(f"The image is expected to have 3 dimensions, found {len(image_array.shape)}")
        elif image_array.shape[-1] != 3:
            raise InvalidShapeError(f"The image is expected to have 3 channels, found {image_array.shape[-1]}")

        # Put the model on eval mode
        self.eval()
        original_shape = image_array.shape
        img_width = find_closest_dividor(initial_number=image_size[0], divisor=32)
        img_height = find_closest_dividor(initial_number=image_size[1], divisor=32)

        # Preprocess the image : transform the image into a tensor while resizing it
        image_tensor = preprocess_input_for_prediction(image=image_array, width=img_width, height=img_height,
                                                       normalize=True)
        # Send the model to the device
        self.to(device)
        with torch.no_grad():
            # Predict the mask
            image_tensor = image_tensor.to(device)
            mask = self.model(image_tensor)
        # Delete the batch dimension
        mask = torch.squeeze(mask, dim=0)
        # Apply an activation function
        if self.model_building_config["classes"] == 1:
            mask = torch.sigmoid(mask)
            mask = torch.round(mask)
        else:
            mask = torch.softmax(mask, dim=0)
            mask = torch.argmax(mask, dim=0)

        # Convert the mask to a numpy array
        mask = torch.squeeze(mask, dim=0).cpu().numpy()
        mask = resize(mask, (original_shape[0], original_shape[1]), preserve_range=True).astype(np.uint8)

        # Post traitement
        if post_process:
            mask = remove_smallest_blobs(mask)
            mask = post_process_mask(mask)

        return mask

    def save(self, path, kwargs=None):
        """Save the model to a file
    This will save informations relative to the model' architecture along with the weights

    Allowed extensions are .pt or .tar

    """
        if kwargs is None:
            kwargs = {}
        if Path(path).suffix in (".pt", ".tar"):
            torch.save({
                "building_config": self.model_building_config,
                "model_state_dict": self.state_dict(),
                **kwargs
            }, path)


def predict_frames_masks(model_checkpoint, images_folder, save_folder_path, image_size: Union[int, List, Tuple] = None,
                        post_process=True, device="cpu"):
    """
    Créér les masques des cadres à partir des annotations YOLOv8
    :param model_checkpoint:
    :param post_process:
    :param device:
    :param image_size:
    :param save_folder_path:
    :param labels_folder:
    :param images_folder:
    :return:
    """
    images_paths = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    model = SegmentationModel(model_checkpoint)

    iterateur = tqdm.tqdm(images_paths, total=len(images_paths))
    for image_path in iterateur:
        new_name = Path(image_path).name.replace(".jpg", ".png")
        save_path = Path(save_folder_path) / new_name
        resultat = model.predict(
            image=image_path,
            image_size=image_size,
            post_process=post_process,
            device=device
        )
        pil_mask = Image.fromarray(resultat*255, mode="L")

        iterateur.set_description("Prédiction du mask pour : " + Path(image_path).name)
        pil_mask.save(save_path, "png")

