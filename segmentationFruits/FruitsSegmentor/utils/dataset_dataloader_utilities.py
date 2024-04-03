from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import albumentations as A

"""Data Documentation"""


def get_train_augs(image_size):
    return A.Compose([
        A.Resize(image_size, image_size)
    ])


def get_val_augs(image_size):
    return A.Compose([
        A.Resize(image_size, image_size)
    ])


class DatasetConfig(object):
    """
    Configuration for loading datasets
    """

    def __init__(self, batch_size: int = 8, image_size: int = 640, augmentations: A.Compose = None,
                 num_classes: int = 3, classes_pixels_values: List[int] = [0, 128, 255],
                 classes_names: List[str] = ["background", "fruit", "edge"],
                 labels_mapping: Dict[int, int] = {0: 0, 128: 1, 255: 2},
                 data=None):
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size
        self.AUGMENTATIONS = augmentations(self.IMAGE_SIZE)
        self.NUM_CLASSES = num_classes
        self.CLASSES_PIXELS_VALUES = classes_pixels_values
        self.CLASSES_NAMES = classes_names
        self.LABELS_MAPPING = labels_mapping

        self.LIST_IMAGES, self.LIST_ANNOTATIONS = data.getData()


"""CUSTOM DATASET"""


class SegmentationDataset(Dataset):

    def __init__(self, dataset_config: DatasetConfig):
        """
    liste_images: liste des chemins absolus vers les images
    liste_annotations; liste des chemins absolus vers les annotations
    classes_pixels_values: valeurs de pixels correspondant aux différentes classes. 0 -> background | 128: fruit | 255: contour

    """
        self.liste_images = dataset_config.LIST_IMAGES
        self.liste_annotations = dataset_config.LIST_ANNOTATIONS
        self.mapping = dataset_config.LABELS_MAPPING
        self.classes_pixels_values = dataset_config.CLASSES_PIXELS_VALUES
        self.augmentations = dataset_config.AUGMENTATIONS
        self.BATCH_SIZE = dataset_config.BATCH_SIZE
        self.IMAGE_SIZE = dataset_config.IMAGE_SIZE

    def __len__(self):
        return len(self.liste_images)

    def mask_to_class_f(self, mask):
        """
    Replaces each pixel with its class index.
    e.g: 0 by 0, 128 by 1 and 255 by 2
    """
        masque = np.copy(mask)
        for k in self.mapping:
            masque[mask == k] = self.mapping[k]
        return masque.astype(dtype=np.int8)

    def __getitem__(self, index):
        image_array = np.asarray(Image.open(self.liste_images[index]))
        f_label_array = np.asarray(Image.open(self.liste_annotations[index]))
        label_array = np.copy(f_label_array)
        label_array = self.mask_to_class_f(label_array)

        if self.augmentations:
            data = self.augmentations(image=image_array, mask=label_array)
            image_array, label_array = data["image"], data["mask"]

        # Transpose to change the dimensions : (height, width, channels) -> (channels, height, width)
        image_array = np.transpose(image_array, (2, 0, 1)).astype(np.float32)
        # label_array = numpy.transpose(label_array, (2, 0, 1)).astype(numpy.int64)

        image_array = torch.Tensor(image_array) / 255.0
        label_array = torch.Tensor(label_array).long()

        return image_array, label_array

    def load_data(self, drop_last=True, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def getDataLoader(dataset_config=None, drop_last=True, shuffle=True, batch_size=None):
    """
    Get the data loader that the trainer and the validator will use to train the model
    """

    dataset = SegmentationDataset(dataset_config)

    return dataset.load_data(drop_last=True, shuffle=True, batch_size=None)
