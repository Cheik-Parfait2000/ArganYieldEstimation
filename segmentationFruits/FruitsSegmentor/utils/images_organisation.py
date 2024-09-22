import csv
import os
from pathlib import Path
import pandas as pd


def getPhotoAttributes(photo_name):
    """
        photo_name : name with/without the extension (e.g. "BIG_1_CadreFruitEst_ArbreEch_2_V1_2022-03-25.jpg")

        return nameWithoutExtension, list_attributes
    """
    nameWithoutExtension = photo_name.split(".")[0]
    name_split = nameWithoutExtension.split("_")

    # List of the photo's attributes : [commune_name, placette_number, photo_orientation, arbre_number,
    # acquisition_date]
    list_attributes = [name_split[0], name_split[1], name_split[2], name_split[4], name_split[6]]

    return nameWithoutExtension, list_attributes


def getImagesAttributes(list_images):
    """
        list_images : list of images names

        return images_attributes: dic {key1: values1, key2: values2}
            key: image name without extension
            values: attributes of the image [commune_name, placette_number, photo_orientation, arbre_number, acquisition_date]
    """

    images_and_attributes = {}

    for image_name in list_images:
        # Récupération du nom et de la liste des attributs
        nameWithoutExtension, list_attributes = getPhotoAttributes(image_name)
        images_and_attributes[f"{nameWithoutExtension}"] = list_attributes

    return images_and_attributes


def writeImagesInfos(images_attributes, file_location, file_name):
    """
        images_attributes : dic {key: value, ...}
            key: image_name
            values: [commune_name, placette_number, photo_orientation, arbre_number, acquisition_date]

        file_location: folder in which to put the newly created file

        file_name: name of the file + its extension
    """

    file = os.path.join(file_location, file_name)
    # Liste du nom des champs
    fieldnames = ["Commune", "Num_placette", "NumArbrEchan", "OrientationImage", "Image", "DateAcquisition", "arbreID"]
    with open(file, "w") as csvFile:
        # Ecriveur : servira à écrire dans le fichier en passant les paramètres sous forme de dictionnaire
        # où les clés sont les noms des champs
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)

        # Ecrire l'entête avec les noms des champs fieldnames'
        writer.writeheader()

        for image_name in images_attributes:
            writer.writerow(
                {
                    "Commune": images_attributes[image_name][0],
                    "Num_placette": images_attributes[image_name][1],
                    "NumArbrEchan": images_attributes[image_name][3],
                    "OrientationImage": images_attributes[image_name][2],
                    "Image": image_name,
                    "DateAcquisition": images_attributes[image_name][4],
                    "arbreID": f"{images_attributes[image_name][3]}-{images_attributes[image_name][0]}_{images_attributes[image_name][1]}"
                }
            )


def get_images_dataframe(images_path):
    """
    Récupérer le dataframe sur les images. Ce dataframe contient les id des arbres, les nom des images, les date
    d'acquisition des images. En réalité, il est la version parsée des noms des images
    :param images_path:
    :return: df
    """
    images_names = os.listdir(images_path)
    images_attributes = getImagesAttributes(images_names)
    writeImagesInfos(images_attributes, os.getcwd(), file_name="DonneesImages.csv")

    file_path = Path(os.getcwd()) / "DonneesImages.csv"
    df = pd.read_csv(file_path)

    os.remove(file_path)

    return df