# ================ Import modules ====================
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

from pathlib import Path

from FruitsSegmentor.utils.errors import InvalidFileError
from FruitsSegmentor.utils.ops import check_yaml, get_unet_config, get_device, build_model_from_dict_config, \
    correct_config_dict_for_model

from typing import Callable, List, Dict, Union, Optional

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

    def predict(self, image: str = None, tilling: bool = False, n_rows: int = None, n_cols: int = None, save_path=None):
        """
        Effectuer une prédiction en utilisant le modèle pré-entrainé.

        image : chemin vers l'image à utilisé
        tilling : spécifie si on doit faire une prédiction en tuillage. Si True, on doit spécifier n_rows et n_cols
        n_rows et n_cols : nombre de lignes et colonnes de la tuile. tuile de 2 * 2 découpe l'image en 4 blocks

        save_path : chemin vers le dossier où enregistrer les prédictions
        """
        pass

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