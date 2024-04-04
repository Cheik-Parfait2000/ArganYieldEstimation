import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from .annotations_utilities import change_path_separator
from PIL import Image


def visualize_image(image, title=""):
    """
    Visualiser une image donnée en utilisant matplotlib
    :param image:
    """
    if isinstance(image, str):
        if Path(image).exists():
            image = change_path_separator(str(Path(image)))
            image = np.asarray(Image.open(image))
            if len(image.shape) == 3:
                if image.shape[-1] == 1:
                    plt.imshow(image, cmap="gray")
                elif image.shape[-1] == 3:
                    plt.imshow(image)
                else:
                    raise NotImplementedError("Can only show images with shape (height, width, channels) with channels \
                                              equal to 1 or 3")
            elif len(image.shape) == 2:
                plt.imshow(image, cmap="gray")

            plt.axis("off")
            plt.title(title)
            plt.show()
        else:
            raise FileNotFoundError("Cannot find the image!")
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            if image.shape[-1] == 1:
                plt.imshow(image, cmap="gray")
            elif image.shape[-1] == 3:
                plt.imshow(image)
            else:
                raise NotImplementedError("Can only show images with shape (height, width, channels) with channels \
                                          equal to 1 or 3")
            plt.axis("off")
            plt.title(title)
            plt.show()
        else:
            raise NotImplementedError("Can only show images with shape (height, width, channels) with channels \
                                                      equal to 1 or 3")
    else:
        raise NotImplementedError("""can only visualize numpy array or path to an image""")
