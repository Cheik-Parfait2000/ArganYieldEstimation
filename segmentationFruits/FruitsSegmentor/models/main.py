# ================ Import modules ====================
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
from skimage.transform import resize

from pathlib import Path

from FruitsSegmentor.utils.dataset_dataloader_utilities import DatasetConfig
from FruitsSegmentor.utils.errors import InvalidFileError, InvalidShapeError
from FruitsSegmentor.utils.ops import check_yaml, get_unet_config, get_device, build_model_from_dict_config, \
    correct_config_dict_for_model, preprocess_input_for_prediction, check_dict_keys_for_train
from FruitsSegmentor.utils.annotations_utilities import tile_image, concat_tiles, find_closest_dividor
from FruitsSegmentor.utils.trainer import Trainer

from typing import Callable, List, Dict, Union, Optional
import numpy as np
from skimage import io
import albumentations as A


DEFAULT_CONFIG = Path(__file__).parent.parent.resolve() / "configs" / "unet_config.pt"
DEFAULT_CONFIG = str(DEFAULT_CONFIG).replace("\\", "/")


# ================= Build the main class ====================
class SegmentationModel(nn.Module):
    """Class de base pour la segmentation d'images

    args:
        model :
            - yaml config file containing the models configuration
            - pretrained_model : .pt file or .tar file
        model_name= Accepted architectures are : unet, unet++ and fpn
    By default, we used a pretrained unet with its weights
    """

    def __init__(self, model: str = DEFAULT_CONFIG, model_name="unet"):
        super().__init__()
        self.model_config = model
        self.model_name = model_name
        self.model = None
        self.model_building_config = None
        self.trainer = None

        # Build the model form the configuration
        self._build_model()
        self.to(get_device())

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

    def fit(self, donnees_dict: dict = None, augmentations: A.Compose = None, batch_size=8, image_size=640,
            save_dir=None, epochs=100, learning_rate=0.005, optimizer=None, lr_scheduler=None, resume_training=True):
        """Train the model
        donnees_dict : the data to use for training :
          - train_images: images and masks for training
          - train_masks: masks for training
          - val_images: images and masks for validation
          - val_masks: masks for validation
        batch_size : the batch size to use for training
        image_size : the image size to use for training
        save_dir : the directory to save the model
        """
        # vérifier la validité des paramètres
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        if not isinstance(donnees_dict, dict):
            raise TypeError("data must be a dictionary")
        else:
            check_dict_keys_for_train(dict_=donnees_dict,
                                      keys=["train_images", "train_masks", "val_images", "val_masks"])
        if augmentations is not None:
            if not isinstance(augmentations, A.Compose):
                raise TypeError("augmentations must be an instance of A.Compose")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if not isinstance(image_size, int):
            raise TypeError("image_size must be an integer")

        if not isinstance(learning_rate, float):
            raise TypeError("Learning rate should be a float")

        if not isinstance(resume_training, bool):
            raise TypeError("resume_training must be a boolean!")

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
        if n_classes == 1:
            classes_pixels_values = [0, 255]
            classes_names = ["background", "Cadre"]
            labels_mapping = {0: 0, 255: 1}
        elif n_classes == 3:
            classes_pixels_values = [0, 128, 255]
            classes_names = ["background", "fruit", "edge"]
            labels_mapping = {0: 0, 128: 1, 255: 2}
        else:
            raise ValueError("Only 1 or 3 classes are supported!")

        train_data_dict = {"images": donnees_dict["train_images"], "labels_masks": donnees_dict["train_masks"]}
        val_data_dict = {"images": donnees_dict["val_images"], "labels_masks": donnees_dict["val_masks"]}
        image_size = find_closest_dividor(initial_number=image_size, divisor=32)
        train_data_cfg = DatasetConfig(batch_size, image_size, augmentations, n_classes,
                                       classes_pixels_values, classes_names, labels_mapping,
                                       data_config=train_data_dict)
        val_data_cfg = DatasetConfig(batch_size, image_size, None, n_classes,
                                     classes_pixels_values, classes_names, labels_mapping,
                                     data_config=val_data_dict)
        # ----------------- Send the task to to trainer --------------- data, model, epochs, save_dir
        self.trainer = Trainer(model=self, _data={"train": train_data_cfg, "val": val_data_cfg})
        self.trainer.train(learning_rate=learning_rate, epochs=epochs, save_dir=save_dir, optimizer=optimizer,
                           lr_scheduler=lr_scheduler, resume_training=resume_training)

    def predict(self, image: Union[str, np.ndarray] = None, tilling: bool = False, tile_size: int = 360,
                prediction_size: int = 640):
        """
        Effectuer une prédiction en utilisant le modèle pré-entrainé.
        image : chemin vers l'image à utilisé ou une image numpy
        tilling : spécifie si on doit faire une prédiction en tuillage. Si True, on doit spécifier n_rows et n_cols
        tile_size: nombre de lignes et colonnes de la tuile. tuile de 2 * 2 découpe l'image en 4 blocks
        prediction_size: taille à laquelle l'image doit être redimensionnée lors pour la prédiction
        """
        if isinstance(image, str):
            if Path(image).exists():
                image = image.replace("\\", "/")
                image_array = io.imread(image)
            else:
                raise FileNotFoundError(f"{image} was not found! please check the good path of your file")
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            raise TypeError("image must be a string or a numpy array")

        # Check for tile, tile_size, prediction_size and save_path
        if isinstance(tilling, bool):
            if tilling:
                if not isinstance(tile_size, int):
                    raise TypeError("tile_size must be an integer")
                if self.model_building_config["classes"] != 3:
                    raise ValueError("Only 3 classes are supported for tuiling!")
        else:
            raise TypeError("tilling must be a boolean")

        if not isinstance(prediction_size, int):
            raise TypeError("prediction_size must be an integer")

        # Check the dimensions of the image : HWC with C=3
        if len(image_array.shape) != 3:
            raise InvalidShapeError(f"The image is expected to have 3 channels, found {image_array.shape[-1]}")
        elif image_array.shape[-1] != 3:
            raise InvalidShapeError(f"The image is expected to have 3 channels, found {image_array.shape[-1]}")

        # Put the model on eval mode
        self.eval()
        original_shape = image_array.shape
        prediction_size = find_closest_dividor(initial_number=prediction_size, divisor=32)
        if tilling:
            # Tile the image :
            tiling_results = tile_image(image=image_array, tile_height=tile_size, tile_width=tile_size)
            tiles, n_rows, n_columns, block_shape = tiling_results.values()
            tile_predictions = []
            for r in range(n_rows):
                for c in range(n_columns):
                    tile = tiles[r, c].reshape(block_shape)
                    # Preprocess the tile for prédiction
                    tile_tensor = preprocess_input_for_prediction(image=tile, width=prediction_size,
                                                                  height=prediction_size, normalize=True)
                    with torch.no_grad():
                        # Predict the mask
                        tile_tensor = tile_tensor.to(get_device())
                        tile_mask = self.model(tile_tensor)
                        tile_mask = torch.squeeze(tile_mask, dim=0)
                        tile_mask = torch.softmax(tile_mask, dim=0)
                        tile_mask = torch.argmax(tile_mask, dim=0)
                        tile_mask = torch.squeeze(tile_mask, dim=0).cpu().numpy()
                        # Resize the tile back to its original size
                        tile_mask = resize(tile_mask, (block_shape[0], block_shape[1]), preserve_range=True).astype(
                            np.uint8)
                        # Add the prediction to the list of predictions
                        tile_predictions.append(tile_mask)
            # Concat tiles predictions
            prediction = concat_tiles(tile_predictions, n_rows, n_columns)
            return prediction
        else:
            # Preprocess the image
            image_tensor = preprocess_input_for_prediction(image=image_array, width=prediction_size,
                                                           height=prediction_size, normalize=True)
            with torch.no_grad():
                # Predict the mask
                image_tensor = image_tensor.to(get_device())
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
            # Convert the mask to a numpy array
            return mask

    def _build_model(self):
        """Build the model from a configuration file or from pretrained weights"""
        if Path(self.model_config).exists():
            if Path(self.model_config).suffix in (".yaml", ".yml"):
                return self._build_model_from_yaml()
            elif Path(self.model_config).suffix in (".pt", ".tar"):
                self._build_pretrained_model(device=get_device())
            else:
                raise InvalidFileError(f"{self.model} is not a recognized file.\n Accepted formats are \
                                       .yml/.yaml/.pt/.tar")
        else:
            raise FileNotFoundError(f"{self.model} was not found! please check the good path of your file")

    def _build_model_from_yaml(self):
        """
        Build the model from yaml configuration file

        if the .yaml/.yml config file is valid, the model will be built
        else:
            raise InvalidConfigFileError("The file self.model_config is not valid")
        """
        if self.model_name == "unet":
            model_config_path = str(Path(self.model_config)).replace("\\", "/")
            if check_yaml(model_architecture="unet", config_file=model_config_path):
                config = get_unet_config(model_config_path)
                self.model_building_config = correct_config_dict_for_model(arch="unet", config_dict=config)
                self.model = build_model_from_dict_config(config=self.model_building_config, architecture="unet")

                # Add a key to specify the models name :
                self.model_building_config["model_name"] = "unet"

                # If there are pretrained weight, restore them
                if Path(config["cpkt_path"]).exists():
                    self._restore_model_weights(config["cpkt_path"], device=get_device())

    def _restore_model_weights(self, weights, device):
        """Restore a model from weights

             weights : path to the weights file or a dict containing the model weights
             device : the device that will receive the content
            """
        try:
            if isinstance(weights, dict):
                self.load_state_dict(weights)
            else:
                self.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        except:
            raise RuntimeWarning("Failed to load the model weights. Please make sure that the weights file is valid.")

    def _build_pretrained_model(self, device):
        """
        Build the model from pretrained weights

        If the model config ends with .pt or .tar, the model will be built

        This file should contain these keys :
          "building_config" : the model configuration. A dict containing the keys
                        "model_name" along with the model architecture keys
          "model_state_dict" : the model weights

        Accepted architectures are : unet, unet++ and fpn

        """
        loaded_config = torch.load(self.model_config, map_location=device)
        self.model_building_config = loaded_config["building_config"]
        self.model_name = self.model_building_config["model_name"]
        arch_config = {
            key: self.model_building_config[key] for key in self.model_building_config
            if key != "model_name"
        }
        if self.model_name == "unet":
            self.model = build_model_from_dict_config(config=arch_config, architecture="unet")
            model_weights = loaded_config["model_state_dict"]
            # Load the model state
            self._restore_model_weights(model_weights, device)
            # self.load_state_dict(model_weights)
        elif self.model_name == "fpn":
            self.model = build_model_from_dict_config(config=arch_config, architecture="fpn")
        elif self.model_name == "unet++":
            self.model = build_model_from_dict_config(config=arch_config, architecture="unet++")

    def __repr__(self):
        """String representation of the model. Keep the default representation provided by Pytorch"""
        return super().__repr__()

    def save(self, path, kwargs=None):
        """Save the model to a file
        This will save informations relative to the model' architecture along with the weights

        Allowed extensions are .pt or .tar or .pth
        """
        if kwargs is None:
            kwargs = {}
        if Path(path).suffix in (".pt", ".tar", ".pth"):
            torch.save({
                "building_config": self.model_building_config,
                "model_state_dict": self.state_dict(),
                **kwargs
            }, path)