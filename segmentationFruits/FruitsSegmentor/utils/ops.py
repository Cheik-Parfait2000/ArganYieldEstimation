import segmentation_models_pytorch as smp
import torch
import numpy as np
from pathlib import Path
from typing import Union

from skimage import io
from skimage import measure

from typing import Callable

from albumentations import Compose, Resize
from albumentations.pytorch.transforms import ToTensorV2
from scipy import ndimage

from .errors import InvalidFileError, InvalidShapeError, InvalidTypeError

ALLOWED_IMAGES_EXTENSIONS = [".jpg", ".jpeg"]


# ================= Quelques fonction utilitaires ==========
def get_device():
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = "cuda"

    return DEVICE


def find_good_length(initial_length: int, total_length: int):
    """
    Find the good length such that the total_length % good_length == 0

    args :
        initial_length : longueur initial
        total_length : la longueur total

    return :
        good_length : la longueur la plus proche de initial_length de telle sorte que
        total_length % good_length == 0
    """
    good_lower_length = 0
    good_higher_length = 0

    for width1 in range(initial_length, 0, -1):
        if total_length % width1 == 0:
            good_lower_length = width1
            break

    for width2 in range(initial_length, total_length + 1, 1):
        if total_length % width2 == 0:
            good_higher_length = width2
            break

    good_length = good_lower_length
    if abs(initial_length - good_higher_length) <= abs(initial_length - good_lower_length):
        good_length = good_higher_length

    return good_length


def find_closest_dividor(initial_number: int, divisor: int):
    """
    Find a number that can be divided by the dividor such that the this number is the closest to initial_number

    args :
        initial_number : le nombre initial à diviser par le diviseur
        divisor : le diviseur

    return :
        good_length : la longueur la plus proche de initial_length de telle sorte que
        total_length % good_length == 0
    """
    good_lower_length = 0
    good_higher_length = 0

    for width1 in range(initial_number, 0, -1):
        if width1 % divisor == 0:
            good_lower_length = width1
            break

    for width2 in range(initial_number, initial_number + divisor + 1, 1):
        if width2 % divisor == 0:
            good_higher_length = width2
            break

    good_length = good_lower_length
    if abs(initial_number - good_higher_length) <= abs(initial_number - good_lower_length):
        good_length = good_higher_length

    return good_length


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

    return torch.Tensor of type torch.float (float32)

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


def check_dict_keys_for_train(dict_, keys):
    """
  Check if the dict contains the keys
  """
    for key in keys:
        if key not in dict_:
            raise KeyError(f"{key} is not in the dict")
        else:
            if not isinstance(dict_[key], str):
                raise TypeError(f"{dict_[key]} must be a path to the data")
            if not Path(dict_[key]).exists():
                raise FileNotFoundError(f"{dict_[key]} is not found! please check your path!")


def remove_smallest_blobs(input_mask):
    """
    Remove all the smaller blobs and only keep the biggest one
    """
    labels_mask = measure.label(input_mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    return labels_mask


def post_process_mask(input_mask):
    """
    Ops :
        - ne retenir que le plus grand blob
        - remplir tous les trous
    """
    mask = remove_smallest_blobs(input_mask)
    mask_final = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return mask_final
