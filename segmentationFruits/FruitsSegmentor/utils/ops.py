import yaml
import segmentation_models_pytorch as smp
import torch
from pathlib import Path



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