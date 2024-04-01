
# ================ Import modules ====================
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

from pathlib import Path 
from utils import (check_yaml, get_unet_config, build_model_from_dict_config, 
                  get_device, correct_config_dict_for_model, InvalidFileError)




# ================= Build the main class ====================

class SegmentationModel(nn.Module):
    """Class de base pour la segmentation d'images

    args:
    model :
        - yaml config file containing the models configuration
        - pretrained_model : .pt file
    type :
        - pretrained
        - config_file
    """
    def __init__(self, model: None, model_name="unet"):
        super().__init__()
        self.model_config = model
        self.model_name = model_name
        self.model = None

        # Build the model form the configuration
        self._build_model()

    def _is_pytorch_model(self):
        pass

    def __call__(self, source):
        """Call directly the model by passing a data to be predicted"""
        pass

    def _build_model(self):
        """Build the model from a configuration file or from pretrained weights"""
        if Path(self.model_config).exists():
            if Path(self.model_config).suffix in (".yaml", ".yml"):
                return self._build_model_from_yaml()
            elif Path(self.model_config).suffix == ".pt":
                self._build_pretrained_model()
            else:
                raise InvalidFileError(f"{self.model} is not a recognized file.\n Accepted formats are .yml/.yaml/.pt")
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
            if check_yaml(model_architecture="unet", config_file=self.model_config):
                config = get_unet_config(self.model_config)
                building_cfg = correct_config_dict_for_model(arch="unet", config_dict=config)
                self.model = build_model_from_dict_config(config=building_cfg, architecture="unet")

            if Path(config["cpkt_path"]).exists():
                self._restore_model_weights(config["cpkt_path"], device=get_device())

    def _restore_model_weights(self, weights_path, device):
        try:
            self.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        except:
            raise RuntimeWarning("Failed to load the model weights. Please make sure that the weights file is valid.")

    def __repr__(self):
        """String representation of the model. Keep the default representation provided by Pytorch"""
        return super().__repr__()