import yaml
import segmentation_models_pytorch as smp
import torch
import numpy as np
from pathlib import Path
from typing import Union
from skimage import io
from typing import Callable
from albumentations import Compose, Resize
from albumentations.pytorch.transforms import ToTensorV2

from .errors import InvalidFileError, InvalidShapeError, InvalidTypeError

ALLOWED_IMAGES_EXTENSIONS = [".jpg", ".jpeg"]


# ================= Quelques fonction utilitaires ==========
def check_yaml(model_architecture="unet", config_file=None):
    """
  Check if the config file is valid for the given architecture.
  We must open the file using the yaml library then check the configuration
  it must contain some attributes.
  """

    def _check_unet_cfg(config_file):
        """Checking for unet"""
        return True

    if model_architecture == "unet":
        return _check_unet_cfg(config_file)


def get_unet_config(path):
    """

  """
    if check_yaml(model_architecture="unet", config_file=path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            model_name = config['architecture_config']['model_name']
            encoder = config['architecture_config']['encoder_name']
            encoder_depth = config['architecture_config']['encoder_depth']
            encoder_weights = config['architecture_config']['encoder_weights']
            decoder_use_batchnorm = config['architecture_config']['decoder_use_batchnorm']
            decoder_attention_type = config['architecture_config']['decoder_attention_type']
            decoder_channels = config['architecture_config']['decoder_channels']
            in_channels = config['architecture_config']['in_channels']
            n_classes = config['architecture_config']['classes']
            activation = config['architecture_config']['activation']
            cpkt_path = config['architecture_config']['checkpoint_path']

            return {
                "model_name": model_name,
                "encoder_name": encoder,
                'encoder_weights': encoder_weights,
                "encoder_depth": encoder_depth,
                "decoder_use_batchnorm": decoder_use_batchnorm,
                "decoder_channels": tuple(decoder_channels),
                "decoder_attention_type": decoder_attention_type,
                "in_channels": in_channels,
                "classes": n_classes,
                "activation": activation,
                "cpkt_path": cpkt_path
            }


def build_model_from_dict_config(config=None, architecture="unet"):
    """Build a unet model from the configuration
  args:
    config: a dictionary containing the keywords along with their values to build
    the model from
    architecture: one of the base architecture available in segmentation-models-pytorch
      "unet", "fpn", "pspnet", etc.
  """
    if architecture == "unet":
        return smp.Unet(**config)
    else:
        raise NotImplementedError("Can only build a model for unet. please implement for others")


def get_device():
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = "cuda"

    return DEVICE


def correct_config_dict_for_model(arch="unet", config_dict=None):
    if arch == "unet":
        correct_attributes_unet = ["encoder_name", "encoder_depth", "encoder_weights", \
                                   "decoder_use_batchnorm", "decoder_channels", \
                                   "decoder_attention_type", "in_channels", "classes", "activation"]
        correct_config = {key: config_dict[key] for key in config_dict if key in correct_attributes_unet}

        correct_config_ = {}
        for key in correct_config:
            if key in ("activation", "decoder_attention_type") and correct_config[key] in ("None", "none"):
                correct_config_[key] = None
            else:
                correct_config_[key] = correct_config[key]

        return correct_config_


def preprocess_input_for_prediction(image: Union[str, np.ndarray, torch.Tensor],
                                    add_batch_dimension: bool = True,
                                    width: int = 640, height: int = 640,
                                    channel_first: bool = False,
                                    normalize: bool = True,
                                    transforms_callback: Callable[..., torch.Tensor] = None) -> torch.Tensor:
    """
    preprocesses an input image to make it ready for prediction

    parameters :
        image : a path to the file or a np.ndarray or a torch.Tensor
        add_batch_dimension :
        width : the output x_size
        height : the output y_size
        channel_first : if the input image is an image or a numpy ndarray, it specifies whether the channel dimension
            is first (channels, rows, cols) or last (rows, cols, channels). Default to False means (rows, cols, channels)
        normalize: whether to normalize the input image or not. Will be normalize by 255. Default to True
        transforms_callback: a  callable that can accept a numpy array and return a torch.Tensor
            with parameters <image_array, x_size, y_size, add_batch_dimension> in this order

    return torch.Tensor

    """
    if isinstance(image, str):
        if Path(image).exists():
            if Path(image).suffix in ALLOWED_IMAGES_EXTENSIONS:
                path = str(Path(image)).replace("\\", "/")
                image_array = io.imread(path, as_gray=False)
            else:
                raise InvalidFileError(f"{image} is not a valid format. \
                                             Accepted formats are {ALLOWED_IMAGES_EXTENSIONS}")
        else:
            raise FileNotFoundError(f"Cannot find the file {image}. Please provide the right path to the image.")
    elif isinstance(image, np.ndarray):
        image_array = image
    elif isinstance(image, torch.Tensor):
        image_array = image.numpy(force=True)

    # Check the dimensions of the image : number of dimensions then number of channels
    if len(image_array.shape) == 3:
        # Check if the number of channels is 3
        if channel_first and image_array.shape[0] != 3:
            raise InvalidShapeError(f"The image is expected to have 3 channels, found {image_array.shape[0]}")
        elif not channel_first:
            if image_array.shape[-1] != 3:
                raise InvalidShapeError(f"The image is expected to have 3 channels, found {image_array.shape[-1]}")
    else:
        raise InvalidShapeError(f"Your image has len(image_array.shape) dimensions. The expected number of dimensions \
                                    is 3")

    if normalize:
        image_array = image_array / 255

    # If a callback is provided, it will be returned, else, the image will be resized
    if transforms_callback is not None:
        return transforms_callback(image_array, width, height, add_batch_dimension)
    else:
        image_tensor = Compose([
            Resize(height, width),
            ToTensorV2(),  # HWC to CHW
        ])(image=image_array)["image"]

    # Add a batch dimension to the image_tensor (channels, rows, cols) -> (1, channels, rows, cols)
    if add_batch_dimension:
        image_tensor = torch.unsqueeze(image_tensor, dim=0)

    return image_tensor.type(torch.float)
