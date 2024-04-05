# ================ Import modules ====================
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
from skimage.transform import resize

from pathlib import Path

from FruitsSegmentor.utils.errors import InvalidFileError
from FruitsSegmentor.utils.ops import check_yaml, get_unet_config, get_device, build_model_from_dict_config, \
    correct_config_dict_for_model, preprocess_input_for_prediction
from FruitsSegmentor.utils.annotations_utilities import tile_image, concat_tiles, find_closest_dividor

from typing import Callable, List, Dict, Union, Optional
import numpy as np
from skimage import io

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
        self.model_building_config = dict()
        self.trainer = None

        # Build the model form the configuration
        self._build_model()
        self.to(get_device())

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

    def save(self, path):
        """Save the model to a file
        This will save informations relative to the model' architecture along with the weights

        Allowed extensions are .pt or .tar

        """
        path_to_save = str(Path(path)).replace("\\", "/")
        if Path(path_to_save).parent.resolve().exists():
            torch.save({
                "building_config": self.model_building_config,
                "model_state_dict": self.state_dict()
            }, path_to_save)
        else:
            raise InvalidFileError(f"The path is incorrect! Cannot find the directory. Correct the path {path}.")