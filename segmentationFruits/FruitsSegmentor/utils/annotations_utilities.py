import os, glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
# from PIL.Image import Image

import skimage
from skimage.util.shape import view_as_blocks
from skimage import io
from skimage.transform import resize

from typing import List, Tuple, Union


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


def tile_image(image: Union[str, JpegImageFile, PngImageFile, Image.Image, np.ndarray],
               tile_height=None, tile_width=None):
    """
    Tuiler une image : la diviser en blocks de n_rows et n_columns
    Les lignes ou colonnes restantes sont ré-attribuées au block les plus proches

    image_name: name of the image with its extension
    n_rows: number of rows
    n_columns: number of columns
    save_path: path to save the image
    """
    if isinstance(image, str):
        if os.path.exists(image):
            image = io.imread(image)
        else:
            raise FileNotFoundError("Can't find the file")

    elif isinstance(image, JpegImageFile | PngImageFile | Image.Image):
        image = np.asarray(image)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError(
            "image must be a str, PIL.JpegImagePlugin.JpegImageFile, PIL.PngImagePlugin.PngImageFile, \
            PIL.Image.Image or np.ndarray")

    # Calcul de la taille de chaque patch

    image_shape = image.shape

    # path_height = image_shape[0] // n_rows
    # patch_width = image_shape[1] // n_columns

    # Find the patch width and height close to the initial values provided.
    patch_width = find_good_length(initial_length=tile_width, total_length=image_shape[1])
    path_height = find_good_length(initial_length=tile_height, total_length=image_shape[0])

    n_rows = image_shape[0] // path_height
    n_columns = image_shape[1] // patch_width

    block_shape = []
    if len(image_shape) == 3:
        block_shape = (path_height, patch_width, image_shape[2])
    else:
        block_shape = (path_height, patch_width)

    # Tuiler l'image en patch : n_rows, n_columns, channels
    patches = view_as_blocks(image, block_shape=block_shape)

    return {"patches": patches, "n_rows": n_rows, "n_columns": n_columns, "block_shape": block_shape}


def concat_tiles(tiles_list, n_rows, n_cols):
    """
      Concatenate tiles

      input:
        tiles_list : list of tiles [t1, t2, t3]
        n_rows : number of rows
        n_cols : number of columns

        n_rows * n_cols = len(tiles_list)
    """
    if len(tiles_list) != n_rows * n_cols:
        raise ValueError(f"Number of tiles must be equal to n_rows*n_columns = {n_rows*n_cols}")

    # Create a list of blocks concatenated horizontally
    h_blocks = [np.hstack(tiles_list[i*n_cols:(i+1)*n_cols]) for i in range(n_rows)]

    # Créér l'image finale en empilant ces blocks horizontal verticalement
    final_image = np.vstack(h_blocks)

    return final_image


class MaskImageConfig(object):
    """
  A configuration class for the mask image.

  We can define its outline width, background color, fill color, and outline color.

  "#000000" -> 0 is the default background color.\n
  "#808080" -> 128 is the default fill color.\n
  "#ffffff" -> 255 is the default outline color.
  """

    def __init__(self, outline_width: int = 3, background_color: str = "#000000",
                 fill_color: str = "#808080", outline_color: str = "#ffffff"):
        """
    Create a mask image configuration.
    Args:
      outline_width: the outline width of the objects in the mask image.
      background_color: the background color of the mask image. For what is not considered as objects
      fill_color: the fill color of the given objects in the mask image.
      outline_color: the outline color of the given objects mask image.

    """
        self.outline_width = outline_width
        self.background_color = background_color
        self.fill_color = fill_color
        self.outline_color = outline_color


"""
=================================== data preparation class ===================================
"""


def change_path_separator(data: str | list, sep_to_replace="\\", new_sep="/"):
    """

    :param data:
    :param sep_to_replace:
    :param new_sep:
    :return:
        data: with the new separator
    """

    def change_single_path_sep(path, old_sep, new_sep_):
        return str(path).replace(old_sep, new_sep_)

    if isinstance(data, str):
        return change_single_path_sep(data, sep_to_replace, new_sep)
    elif isinstance(data, list):
        return list(map(lambda e: change_single_path_sep(e, sep_to_replace, new_sep), data))
    else:
        raise NotImplementedError("Can only modify for str paths or list of str paths")


class DataPreparator(object):
    """
        A class to prepare data for training and validation
        It is characterised by its folder where we can find images and annotations in a specific format
    """

    def __init__(self, data_folder: str, mask_annotation_config: MaskImageConfig = None, data_format: str = "yolov8"):
        self.data_format = data_format
        self.data_folder = data_folder
        self.images = None
        self.annotations = None
        self.mask_annotation_config = mask_annotation_config

    def get_images_annotations(self):
        """
          Get the images and annotations
        """
        if self.data_format == "yolov8":
            self.images = list(map(str, (Path(self.data_folder) / "images").glob("*jpg")))
            self.annotations = list(map(str, (Path(self.data_folder) / "labels").glob("*.txt")))

            return self.images, self.annotations
        else:
            raise NotImplementedError

    def get_polygons(self, path_to_label_file: str, image_path: str):
        """
          Permet de récupérer les polygones présents dans un fichier donnée:
          return:
            polygones: [[(x, y), (x, y), (x, y)], ...]
        """
        if os.path.exists(path_to_label_file):
            if self.data_format == "yolov8":
                with open(path_to_label_file) as file:
                    content = file.readlines()
                    if len(content) > 0:
                        polygones = []
                        imArray = np.asarray(Image.open(image_path).convert("RGBA"))
                        width, height = imArray.shape[0], imArray.shape[1]
                        for e in content:
                            pol = [float(i) for i in e.split(" ")[1:]]
                            pol = [(int(pol[j] * width), int(pol[j + 1] * height)) for j in range(0, len(pol), 2)]
                            polygones.append(pol)

                        return polygones
            else:
                raise NotImplementedError("The data format is not supported")

    def create_img_mask(self, img_path, polygones_path):
        """
            Créer un mask d'image à partir d'une image et des ses annotations en polygone

            img_path : chemin de l'image
            polygones_path: chemin des polygones de l'image

            background_color: the background color of the image
            fill_color: couleur de remplissage du polygone
            outline_color: couleur de la ligne de contour du polygone
            outline_width: epaisseur de la ligne de contour
        """

        # Récupération des polygones correspondant à l'image
        data_polygones = self.get_polygons(polygones_path, img_path)

        # Convertire l'image en numpy_array
        imArray = np.asarray(Image.open(img_path).convert("RGBA"))

        # Mask image to be created characteristics
        outline_width = self.mask_annotation_config.outline_width
        background_color = self.mask_annotation_config.background_color
        fill_color = self.mask_annotation_config.fill_color
        outline_color = self.mask_annotation_config.outline_color

        # Création du masque:
        # mode=L (8-bit pixels, grayscale)
        maskImage = Image.new(mode='L', size=(imArray.shape[1], imArray.shape[0]), color=background_color)
        for pol in data_polygones:  # Dessiner chaque polygone sur l'image
            ImageDraw.Draw(maskImage).polygon(pol, outline=outline_color, fill=fill_color, width=outline_width)
            ImageDraw.Draw(maskImage).polygon(pol, outline=outline_color, fill=fill_color, width=outline_width)

        return maskImage

    def create_mask_annotations(self, tile: bool = True, tile_height=256, tile_width=256,
                                classes_to_satisfy: list = [128],
                                save_path=None, sub_folder=None):
        """
            Create masks from polygon annotations
            tile: if True, the images will be tiled
            tile_height: the width of the tile
            tile_width: the heigth of each tile.
                Note : tile_height and tile_width can be modified in the process
                if the width % tile_width and/or heigth % tile_height are not equal to 0.
            classes_to_satisfy : Classes values that must be in the tiles. Otherwise the tile masks with their
                corresponding image patch won't be saved
            save_path: path to save the masks
            sub_folder: sub folder to save the data in case the data is tiled
        """
        if save_path is None:
            save_path = self.data_folder
        if self.data_format == "yolov8":
            images, annotations = self.get_images_annotations()
            images, annotations = change_path_separator(images), change_path_separator(annotations)
            if len(images) > 0 and len(annotations) > 0:

                # Gestion des dossiers pour le stockage des données
                if tile:
                    if sub_folder is None and save_path == self.data_folder:
                        raise Exception("The save path is the same as the data folder. you must enter a sub_folder!")
                    elif save_path != self.data_folder and sub_folder is not None:
                        Path(save_path).mkdir(exist_ok=True)
                        (Path(save_path) / sub_folder).mkdir(exist_ok=True)
                        (Path(save_path) / sub_folder / "images").mkdir(exist_ok=True)
                        (Path(save_path) / sub_folder / "labels_masks").mkdir(exist_ok=True)

                        save_path = str(Path(save_path) / sub_folder)
                    else:
                        Path(save_path).mkdir(exist_ok=True)
                        save_path = str(Path(save_path) / sub_folder)
                else:
                    (Path(save_path) / "labels_masks").mkdir(exist_ok=True)
                    save_path = str(Path(save_path) / "labels_masks")
                loop = tqdm(enumerate(images), total=len(images), desc=f"Saving data to -> {save_path}...")
                for idx, image_path in loop:
                    image_name_sans_extension = image_path.split("/")[-1][:-4]
                    label_path = os.path.join(self.data_folder, f"labels/{image_name_sans_extension}.txt").replace("\\", "/")
                    # Si l'image a une annotation et que cette annotation n'est pas vide
                    if label_path in annotations and self.get_polygons(label_path, image_path):
                        mask_pil_image = self.create_img_mask(image_path, label_path)
                        if tile:
                            # Create the save path for images and labels_masks
                            save_images_path = str(Path(save_path) / "images")
                            save_masks_path = str(Path(save_path) / "labels_masks")

                            # Tile images and save them in their paths
                            image_name = image_name_sans_extension + ".png"
                            patches_sans_annotation = self.tile_image(mask_pil_image, image_name=image_name,
                                                                      tile_height=tile_height,
                                                                      tile_width=tile_width, save_path=save_masks_path,
                                                                      classes_to_satisfy=classes_to_satisfy)
                            self.tile_image(image_path, image_name=None, tile_height=tile_height, tile_width=tile_width,
                                            save_path=save_images_path, patches_to_discard=patches_sans_annotation)

                        else:
                            path_to_save = os.path.join(save_path, f"{image_name_sans_extension}.png")
                            mask_pil_image.save(path_to_save)
                    else:
                        loop.set_postfix(skip_for=f"{image_name_sans_extension}")

    def tile_image(self, image: Union[
        str, JpegImageFile, PngImageFile, Image.Image, np.ndarray],
                   image_name=None, tile_height=None, tile_width=None, save_path=None,
                   classes_to_satisfy: List[int] = [], patches_to_discard: List[str] = None):
        """
        Tuiler une image : la diviser en blocks de n_rows et n_columns
        Les lignes ou colonnes restantes sont ré-attribuées au block les plus proches

        image_name: name of the image with its extension
        n_rows: number of rows
        n_columns: number of columns
        save_path: path to save the image
        """
        save_file_name = ""
        if isinstance(image, str):
            if os.path.exists(image):
                save_file_name = image
            else:
                raise FileNotFoundError("Can't find the file")

        elif isinstance(image, JpegImageFile | PngImageFile | Image.Image):
            if image_name is None:
                raise ValueError("image_name must be defined")
            save_file_name = image_name

        elif isinstance(image, np.ndarray):
            if image_name is None:
                raise ValueError("image_name must be defined")
            save_file_name = image_name
        else:
            raise TypeError(
                "image must be a str, PIL.JpegImagePlugin.JpegImageFile, PIL.PngImagePlugin.PngImageFile, \
                PIL.Image.Image or np.ndarray")
        if isinstance(classes_to_satisfy, list):
            for e in classes_to_satisfy:
                if not isinstance(e, int):
                    raise ValueError("Classes to satisfy must be a list of integers")
        else:
            raise ValueError("Classes to satisfy must be a list of integers")

        # Tuiler l'image en patch : n_rows, n_columns, channels
        tiling_result = tile_image(image=image, tile_height=tile_height, tile_width=tile_width)

        n_rows = tiling_result["n_rows"]
        n_columns = tiling_result["n_columns"]
        patches = tiling_result["patches"]
        block_shape = tiling_result["block_shape"]

        if save_path is not None:
            # Patches that do not contain some classes
            non_valid_patches = []
            for r in range(n_rows):
                for c in range(n_columns):
                    patch_to_save = patches[r, c].reshape(block_shape)
                    file_name = save_file_name.split("/")[-1][:-4] + f"_row{r}_column{c}" + '.' + \
                                save_file_name.split('.')[-1]
                    if len(classes_to_satisfy) > 0:
                        uniques_classes = np.unique(patch_to_save).tolist()
                        save_patch = True
                        for cls in classes_to_satisfy:
                            if cls not in uniques_classes:
                                save_patch = False
                                non_valid_patches.append(file_name[:-4])
                                break
                        if save_patch:
                            file_path = os.path.join(save_path, file_name).replace("\\", "/")
                            io.imsave(file_path, patch_to_save,
                                      check_contrast=False)
                    else:
                        if file_name[:-4] not in patches_to_discard:
                            file_path = os.path.join(save_path, file_name).replace("\\", "/")
                            io.imsave(file_path, patch_to_save,
                                      check_contrast=False)
            return non_valid_patches
        else:
            patchs = []
            for r in range(n_rows):
                for c in range(n_columns):
                    patchs.append(patches[r, c].reshape(block_shape))

            return patchs

