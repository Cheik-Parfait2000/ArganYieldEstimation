import os, glob, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageColor
from skimage.measure import find_contours
from tqdm import tqdm

from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
# from PIL.Image import Image

import skimage
from skimage.util.shape import view_as_blocks
from skimage import io
from skimage.transform import resize

from typing import List, Tuple, Union

from .ops import find_good_length, find_closest_dividor


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
    """

    def __init__(self, classes_outline_width: int = 0,
                 n_classes: int = 3):
        """
        Create a mask image configuration.
        background_color: str = "#000000",
                  fill_color: str = "#808080", outline_color: str = "#ffffff"
        Args:
          classes_outline_width: The outline width of the classes. Defaults to 0.
          n_classes: The number of classes in the mask image without the background class
        """
        self.background_color = "black"
        self.classes_colors = None
        self.classes_outline_width = classes_outline_width
        self.classes_outline_color = None
        self.n_classes = n_classes

        self.check_config()

    def choose_colors(self, n_classes, exclude_colors: list = None):
        """
          Choisir les couleurs de manière aléatoire pour chaque classe
          n_classes: nombre de classes
          exclude_colors: couleurs à exclure

          return:
            colors: couleurs choisies aléatoirement et leurs valeur en niveau de gris
        """
        couleurs = ImageColor.colormap
        colors = list(couleurs.keys())
        if exclude_colors is not None:
            for color in exclude_colors:
                colors.remove(color)

        random_indexes = np.random.randint(0, len(colors), size=n_classes).tolist()
        colors = [colors[i] for i in random_indexes]
        # colors_grayscale_values = [ImageColor.getcolor(color, "L") for color in colors]

        return colors

    def check_config(self):
        """
        Check the configuration of the mask image.
        """
        couleurs_a_exclure = [self.background_color]

        if not isinstance(self.n_classes, int):
            raise TypeError("n_classes must be an integer")

        if self.n_classes < 1:
            raise ValueError("n_classes must be greater than  or equal to 1")

        if self.classes_outline_width < 0:
            raise ValueError("classes_outline_width must be greater than or equal to 0")

        if self.classes_outline_width > 0:
            self.classes_outline_color = self.choose_colors(1)
            couleurs_a_exclure.append(self.classes_outline_color)

        self.classes_colors = self.choose_colors(self.n_classes, couleurs_a_exclure)


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
        self.classes_mapping_to_colors = dict()

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
                        classes = []
                        imArray = np.asarray(Image.open(image_path).convert("RGBA"))
                        width, height = imArray.shape[1], imArray.shape[0]
                        for e in content:
                            classe = int(e.split(" ")[0])
                            classes.append(classe)
                            pol = [float(i) for i in e.split(" ")[1:]]
                            pol = [(int(pol[j] * width), int(pol[j + 1] * height)) for j in range(0, len(pol), 2)]
                            polygones.append(pol)

                        return polygones, classes
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
        data_polygones, classes = self.get_polygons(polygones_path, img_path)

        # Convertire l'image en numpy_array
        imArray = np.asarray(Image.open(img_path).convert("RGBA"))

        # Mask image to be created characteristics
        background_color = self.mask_annotation_config.background_color
        outline_width = self.mask_annotation_config.classes_outline_width
        outline_color = self.mask_annotation_config.classes_outline_color

        # Création du masque:
        # mode=L (8-bit pixels, grayscale)
        maskImage = Image.new(mode='L', size=(imArray.shape[1], imArray.shape[0]), color=background_color)

        for pol, classe in zip(data_polygones, classes):  # Dessiner chaque polygone sur l'image
            fill_color = self.mask_annotation_config.classes_colors[classes[classe]]
            ImageDraw.Draw(maskImage).polygon(pol, outline=outline_color, fill=fill_color, width=outline_width)

            # Ajouter le numéro de la classe, sa couleur et la valeur correspondante en niveau de gris
            # dans le mapping des classes aux couleurs
            self.classes_mapping_to_colors[classe] = [fill_color, ImageColor.getcolor(fill_color, "L")]

        return maskImage

    def create_mask_annotations(self, save_path=None, sub_folder=None):
        """
            Create masks from polygon annotations
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
                Path(save_path).mkdir(exist_ok=True)
                if isinstance(sub_folder, str):
                    (Path(save_path) / sub_folder).mkdir(exist_ok=True)
                    save_path = str(Path(save_path) / sub_folder)
                else:
                    (Path(save_path) / "labels_masks").mkdir(exist_ok=True)
                    save_path = str(Path(save_path) / "labels_masks")
                loop = tqdm(enumerate(images), total=len(images), desc=f"Saving data to -> {save_path}...")
                for idx, image_path in loop:
                    image_name_sans_extension = image_path.split("/")[-1][:-4]
                    label_path = os.path.join(self.data_folder, f"labels/{image_name_sans_extension}.txt").replace("\\",
                                                                                                                   "/")
                    # Si l'image a une annotation et que cette annotation n'est pas vide
                    if label_path in annotations and self.get_polygons(label_path, image_path):
                        mask_pil_image = self.create_img_mask(image_path, label_path)
                        path_to_save = os.path.join(save_path, f"{image_name_sans_extension}.png")
                        mask_pil_image.save(path_to_save)
                    else:
                        loop.set_postfix(skip_for=f"{image_name_sans_extension}")

        # Sauvegarder le fichier json de mappage des classes
        save_file_path = str(Path(save_path) / "classes_colors_mapping.json").replace("\\", "/")
        self.classes_mapping_to_colors["background_color"] = [self.mask_annotation_config.background_color,
                                                              ImageColor.getcolor(
                                                                  self.mask_annotation_config.background_color, "L")]
        if self.mask_annotation_config.classes_outline_color is not None:
            self.classes_mapping_to_colors["outline_color"] = [self.mask_annotation_config.classes_outline_color,
                                                               ImageColor.getcolor(
                                                                   self.mask_annotation_config.classes_outline_color,
                                                                   "L")]
        self.classes_mapping_to_colors["outline_width"] = self.mask_annotation_config.classes_outline_width
        with open(save_file_path, "w") as f:
            json.dump(self.classes_mapping_to_colors, f)


# ================== Convert mask annotation to yolov8 =========================
def mask_annotation_to_yolov8(mask_annotation_path, reassignment_mapping={255: 0}):
    """
    Convert mask annotation to yolov8 format

    args:
    mask_annotation_path: path to the mask annotation
    save_path: path to save the converted mask annotation
    reassignment_mapping: dictionary of reassignment mapping

    return:
    objects_coordinates: list of objects coordinates that are normalized to the image shape
    """

    if isinstance(mask_annotation_path, str):
        if not Path(mask_annotation_path).exists():
            raise FileNotFoundError("Can't find the file")

    image = np.array(Image.open(mask_annotation_path))
    image_shape = image.shape
    if len(reassignment_mapping.keys()) > 0:
        for key in reassignment_mapping.keys():
            image[image == key] = reassignment_mapping[key]
    contours = find_contours(image)
    objects_coordinates = []
    for ctr in contours:
        rows = (ctr[:, 0] / image_shape[0]).tolist() # y
        cols = (ctr[:, 1] / image_shape[1]).tolist() # x
        obj = []
        for col, row in zip(cols, rows):
            obj.append(col)
            obj.append(row)
        # If the contour is not closed, close it
        if obj[0] != obj[-2] and obj[1] != obj[-1]:
            obj.append(obj[0])
            obj.append(obj[1])
        objects_coordinates.append(obj)
    return objects_coordinates


def create_yolov8_annotations_from_masks(images_folder, save_folder, reassignment_mapping={255: 0}, class_value=0):
    """
    Create yolov8 annotations from masks and save them in a folder

    args:
    images_folder: path to the images folder
    save_folder: path to save the annotations
    reassignment_mapping: dictionary of reassignment mapping

    return:
    None
    """
    if not Path(images_folder).exists():
        raise FileNotFoundError("Can't find the images folder")

    Path(save_folder).mkdir(exist_ok=True)
    list_images = list(Path(images_folder).glob("*.png"))
    loop = tqdm(list_images, total=len(list_images), desc=f"Saving annotations to :: {save_folder}")
    for file in loop:
        mask_path = str(file).replace("\\", "/")
        save_path = str(Path(save_folder) / file.name).replace("\\", "/")[:-4] + ".txt"
        objects_coordinates = mask_annotation_to_yolov8(mask_path, reassignment_mapping)
        with open(save_path, "w") as f:
            for obj in objects_coordinates:
                row = f"{class_value}"
                for e in obj:
                    row += f" {e}"
                    row += "\n"
                f.write(row)