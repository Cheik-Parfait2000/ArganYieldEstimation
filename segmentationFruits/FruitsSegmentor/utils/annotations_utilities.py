import os, glob
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import skimage
from skimage.util.shape import view_as_blocks
from skimage import io
from skimage.transform import resize

from typing import List, Tuple


class MaskImageConfig(object):
    """
    A configuration class for the mask image.

    We can define its outline width, background color, fill color, and outline color.

    "#000000" -> 0 is the default background color.
    "#808080" -> 128 is the default fill color.
    "#ffffff" -> 255 is the default outline color.
    """
    def __init__(self):
        self.outline_width: int = 3
        self.background_color: str = "#000000"
        self.fill_color: str = "#808080"
        self.outline_color: str = "#ffffff"


class Data(object):
    """
        A class representing annotated images
        It is characterised by its folder where we can find images and annotations in a specific format
    """
    def __init__(self, data_folder: str, mask_annotation_config: MaskImageConfig = MaskImageConfig(),\
        masks_folder_name: str = "labels_masks", tiled_data_folder_name: str = "tuiles", data_format: str = "yolov8"):
        self.data_format = data_format
        self.data_folder = data_folder
        self.images = None
        self.annotations = None
        self.mask_annotation_config = mask_annotation_config
        self.masks_folder_name = masks_folder_name
        self.tiled_data_folder_name = tiled_data_folder_name

    def get_images_annotations(self):
        """
        Get the images and annotations
        """
        if self.data_format == "yolov8":
            self.images = glob.glob(os.path.join(self.data_folder, "images")+"/*.jpg")
            self.annotations = glob.glob(os.path.join(self.data_folder, "labels")+"/*.txt")

            return self.images, self.annotations
        else:
            raise NotImplementedError(f"Cannot create annotations for {self.data_format}. Please implement that section here.")


    def get_polygons(self, path_to_label_file: str, image_path: str):
        """
        Permet de récupérer les polygones présents dans un fichier donnée:
        return:
        polygones: [[(x, y), (x, y), (x, y)], ...]
        """
        if Path(path_to_label_file).exists():
            if self.data_format == "yolov8":
                with open(path_to_label_file) as file:
                    content = file.readlines()
                    if len(content) > 0:
                        polygones = []
                        imArray = np.asarray(Image.open(image_path).convert("RGBA"))
                        width, height = imArray.shape[0], imArray.shape[1]
                        for e in content:
                            pol = [float(i) for i in e.split(" ")[1:]]
                            pol = [(int(pol[j]*width), int(pol[j+1]*height)) for j in range(0, len(pol), 2)]
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
        for pol in data_polygones: # Dessiner chaque polygone sur l'image
            ImageDraw.Draw(maskImage).polygon(pol, outline=outline_color, fill=fill_color, width=outline_width)
            ImageDraw.Draw(maskImage).polygon(pol, outline=outline_color, fill=fill_color, width=outline_width)

        return maskImage

    def create_mask_annotations(self):
        """
        Create masks from polygon annotations
        """
        if self.data_format == "yolov8":
            images, annotations = self.get_images_annotations()
            if len(images) > 0 and len(annotations) > 0:
                if not os.path.exists(os.path.join(self.data_folder, "labels_masks")):
                    # Créér un sous dossier pour sauvegarder les masks
                    os.mkdir(os.path.join(self.data_folder, "labels_masks"))

                loop = tqdm(enumerate(images), total=len(images), desc="creating annotations ...")
                for idx, image_path in loop:
                    image_name_sans_extension = image_path.split("/")[-1][:-4]
                    label_path = os.path.join(self.data_folder, f"labels/{image_name_sans_extension}.txt")

                    loop.set_postfix(processing_for = f"{image_name_sans_extension}")

                    # Si l'image a une annotation et que cette annotation n'est pas vide
                    if label_path in annotations and self.get_polygons(label_path, image_path):
                        path_to_save = os.path.join(self.data_folder, f"labels_masks/{image_name_sans_extension}.png")
                        mask_pil_image = self.create_img_mask(image_path, label_path)
                        mask_pil_image.save(path_to_save)
                    else:
                        loop.set_postfix(skip_for = f"{image_name_sans_extension}")
        else:
           raise NotImplementedError("Can only process yolov8 annotation format")

    def create_tiled_data(self, n_rows=3, n_columns=3):
        """
        Create tiled data : will tile the images and masks into parts for training or validation
        """
        save_base_dir = os.path.join(self.data_folder, self.tiled_data_folder_name)
        save_imgs_path = os.path.join(save_base_dir, "images")
        save_masks_path = os.path.join(save_base_dir, "masks")

        # Create folders for the tiles
        if not os.path.exists(save_base_dir):
            os.mkdir(save_base_dir)
        if not os.path.exists(save_imgs_path):
            os.mkdir(save_imgs_path)
        if not os.path.exists(save_masks_path):
            os.mkdir(save_masks_path)

        images_path = os.path.join(self.data_folder, "images")
        annotations_path = os.path.join(self.data_folder, "labels_masks")

        paths = [images_path, annotations_path]
        save_paths = [save_imgs_path, save_masks_path]
        images_formats = [".jpg", ".png"]

        for images_path, save_path, images_format in zip(paths, save_paths, images_formats):
            loop = tqdm(glob.glob(os.path.join(images_path, "*"+images_format)), total = len(glob.glob(os.path.join(images_path, "*"+images_format))))
            for img_path in loop:
                loop.set_description(f"Processing images of : {images_path}")
                self.tile_image(img_path, n_rows=3, n_columns=3, save_path=save_path)



    def tile_image(self, image_path, n_rows, n_columns, save_path=None):
        """
        Tuiler une image : la diviser en blocks de n_rows et n_columns
        Les lignes ou colonnes restantes sont ré-attribuées au block les plus proches
        """
        if os.path.exists(image_path):
            image = io.imread(image_path)

            # Calcul de la taille de chaque patch

            image_shape = image.shape

            path_height = image_shape[0] // n_rows
            patch_width = image_shape[1] // n_columns

            remainder_rows = image_shape[0] % n_rows
            remainder_columns = image_shape[1] % n_columns

            block_shape = []
            if len(image_shape) == 3:
                block_shape = (path_height, patch_width, image_shape[2])
            else:
                block_shape = (path_height, patch_width)

            # Tuiler l'image en pach : n_rows, n_columns, channels
            patches = view_as_blocks(image, block_shape=block_shape)

            if save_path != None:
                for r in range(n_rows):
                    for c in range(n_columns):
                        file_name = image_path.split("/")[-1][:-4] + f"_row{r}_column{c}" + '.' + image_path.split('.')[-1]
                        io.imsave(os.path.join(save_path, file_name), patches[r, c].reshape(block_shape))
            else:
                patchs = []
                for r in range(n_rows):
                    for c in range(n_columns):
                        patchs.append(patches[r, c].reshape(block_shape))

                return patchs
        else:
            raise FileNotFoundError("Can't find the file")

    def getData(self, type: str="tuiles"):
        """
        Get the list of data : images and masks
        to build a dataset
        """
        if type == "tuiles":
            if os.path.exists(self.data_folder):
                images_paths = glob.glob(os.path.join(self.data_folder, "tuiles", "images")+"/*.jpg")
                labels_paths = glob.glob(os.path.join(self.data_folder, "tuiles", "masks")+"/*.png")

                liste_images = []
                liste_annotations = []
                for img_path in images_paths:
                    label_path = os.path.join(self.data_folder, "tuiles", "masks", img_path.split("/")[-1][:-4]+".png")
                    if label_path in labels_paths:
                        liste_images.append(img_path)
                        liste_annotations.append(label_path)

                return sorted(liste_images), sorted(liste_annotations)
        else:
            raise NotImplementedError("Not implemented")
