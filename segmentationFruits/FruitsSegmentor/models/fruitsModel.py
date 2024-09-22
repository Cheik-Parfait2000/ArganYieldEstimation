import glob
import math
import random
from pathlib import Path

import cv2
import numpy as np
import tqdm
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections
from skimage import io
from skimage.measure import find_contours, label, regionprops
from ultralytics import YOLO
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from FruitsSegmentor.models import SegmentationModel
from FruitsSegmentor.utils.ops import get_device
from FruitsSegmentor.utils.images_organisation import get_images_dataframe

import logging
logging.disable(logging.WARNING)


# ==================================== Fonctions ========================================
def save_content(liste_data, path):
    """
    Sauvegarder une liste de float au format txt
    """
    content = ",".join([str(e) for e in liste_data])
    with open(path, "w") as f:
        f.writelines(content)


def get_polygons(path_to_label_file: str, image_path: str):
    """
          Permet de récupérer les polygones présents dans un fichier donnée:
          return:
            polygones: [[(x, y), (x, y), (x, y)], ...]
        """
    if os.path.exists(path_to_label_file):
        with open(path_to_label_file) as file:
            content = file.readlines()
            if len(content) > 0:
                polygones = []
                W, H = Image.open(image_path).convert("RGB").size
                for e in content:
                    pol = [float(i) for i in e.split(" ")[1:]]
                    pol = [(pol[j] * W, pol[j + 1] * H) for j in range(0, len(pol), 2)]
                    # pol = [(pol[j], pol[j + 1]) for j in range(0, len(pol), 2)]
                    polygones.append(pol)

                return polygones


def create_frame_mask(path_to_label_file, image_path):
    """
    Créér un masque png du cadre à partir de son annotation
    :param path_to_label_file:
    :param image_path:
    :return:
    """
    polygons = get_polygons(path_to_label_file, image_path)
    image = Image.open(image_path)

    mask = Image.new("L", image.size, 0)
    if polygons is not None:
        for pol in polygons:
            if len(pol) >= 2:
                ImageDraw.Draw(mask).polygon(pol, fill="white", outline=0, width=0)

        return mask


def create_frames_masks(labels_folder, images_folder, save_folder_path):
    """
    Créér les masques des cadres à partir des annotations YOLOv8
    :param save_folder_path:
    :param labels_folder:
    :param images_folder:
    :return:
    """
    labels_paths = sorted(glob.glob(os.path.join(labels_folder, "*.txt")))
    images_paths = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))

    iterateur = tqdm.tqdm(zip(labels_paths, images_paths), total=len(labels_paths))
    for label_path, image_path in iterateur:
        new_name = Path(image_path).name.replace(".jpg", ".png")
        save_path = Path(save_folder_path) / new_name
        pil_mask = create_frame_mask(label_path, image_path)
        if pil_mask is None:
            iterateur.set_description("Aucun label pour " + Path(image_path).name)
        else:
            iterateur.set_description("Création du mask pour : " + Path(image_path).name)
            pil_mask.save(save_path, "png")


def polygon_to_bbox(polygon):
    x_values = [e[0] for e in polygon]
    y_values = [e[1] for e in polygon]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    return x_min, x_max, y_min, y_max


def bbox_to_point(bounding_box):
    """
    bbox:
        x_min, x_max, y_min, y_max

    """
    width = (bounding_box[1] - bounding_box[0]) / 2
    height = (bounding_box[3] - bounding_box[2]) / 2
    column = bounding_box[0] + width
    row = bounding_box[2] + height
    column = int(column)
    row = int(row)
    return column, row


def bboxes_to_points(bounding_boxes):
    """
    bounding_boxes : [[xmin, ymin, xmax, ymax]] bboxes_to_points(pred_boxes)
    """
    points = []
    for bbox__ in bounding_boxes:
        bbox__ = [bbox__[0], bbox__[2], bbox__[1], bbox__[3]]
        x__, y__ = bbox_to_point(bbox__)
        points.append((x__, y__))
    return points


def polygons_to_bboxes(polygons):
    """
    polygons:
        [
            [(x, y), (x, y), ...],
            .....................,
            [(x, y), (x, y), ...]
        ]
    """
    bboxes = []
    for polygon in polygons:
        bbox = polygon_to_bbox(polygon)
        bbox = [bbox[0], bbox[2], bbox[1], bbox[3]]
        bboxes.append(bbox)

    return bboxes


def polygons_to_points(polygons):
    """
    polygons:
        [
            [(x, y), (x, y), ...],
            .....................,
            [(x, y), (x, y), ...]
        ]

    return points
    """
    points = []
    for polygon in polygons:
        bbox = polygon_to_bbox(polygon)
        point = bbox_to_point(bbox)
        points.append((point[0], point[1]))

    return points


def get_label_points(path_to_label_file, image_path):
    """
    Convertir les coordonnées des fruits annotées en points
    """
    polygons = get_polygons(path_to_label_file, image_path)
    points = polygons_to_points(polygons)
    return points


def create_points_mask(img_path, polygones_path):
    """
            Créer un mask d'image à partir d'une image et des ses annotations en polygone

            img_path : chemin de l'image
            polygones_path: chemin des polygones de l'image

            background_color: the background color of the image
            fill_color: couleur de remplissage du polygone
            outline_color: couleur de la ligne de contour du polygone
            outline_width: epaisseur de la ligne de contour
    """
    # Récupération des points correspondant à l'image
    points = get_label_points(polygones_path, img_path)

    # Convertire l'image en numpy_array
    imArray = np.asarray(Image.open(img_path))

    # Création du masque:
    # mode=L (8-bit pixels, grayscale)
    maskImage = Image.new(mode='L', size=(imArray.shape[1], imArray.shape[0]), color="black")
    ImageDraw.Draw(maskImage).point(points, fill="white")

    return maskImage


def yolo_pred_to_points(bbox_list):
    """
    bbox : xyxy
    """

    points = []
    for r in bbox_list:
        bbox = (r[0], r[2], r[1], r[3])
        point = bbox_to_point(bbox)
        points.append(point[0])
        points.append(point[1])
    return points


def slice_pred_to_points(prediction):
    points = []
    for r in prediction.object_prediction_list:
        bbox = (r.bbox.minx, r.bbox.maxx, r.bbox.miny, r.bbox.maxy)
        point = bbox_to_point(bbox)
        points.append(point[0])
        points.append(point[1])
    return points


def yolo_pred_to_points_mask(img_path, prediction):
    # Récupération des points correspondant à l'image
    points = yolo_pred_to_points(prediction)

    # Convertire l'image en numpy_array
    imArray = np.asarray(Image.open(img_path))

    # Création du masque:
    # mode=L (8-bit pixels, grayscale)
    maskImage = Image.new(mode='L', size=(imArray.shape[1], imArray.shape[0]), color="black")
    ImageDraw.Draw(maskImage).point(points, fill="white")

    return maskImage


def get_true_predicted_number(img_path, mask_cadre_path, prediction):
    points_mask = np.asarray(yolo_pred_to_points_mask(img_path, prediction))
    mask_cadre = np.asarray(Image.open(mask_cadre_path))

    # multiply to keep only points in the frame
    product_mask = np.multiply(points_mask, mask_cadre)
    n_fruits = len(find_contours(product_mask))

    return n_fruits


def get_True_number_fruits(img_path, label_path, mask_cadre_path):
    """

    """
    points_mask = np.asarray(create_points_mask(img_path, label_path))
    mask_cadre = np.asarray(Image.open(mask_cadre_path))

    # multiply to keep only points in the frame
    product_mask = np.multiply(points_mask, mask_cadre)
    n_fruits = len(find_contours(product_mask))

    return n_fruits


def pols_to_mask(polygons, img, edge_width=3, fill_color="white", outline="black"):
    """
    polygons : [[(x, y), (x, y), ...]]
    Créer un mask des fruits à partir des polygones
    """
    image = img
    if isinstance(img, str):
        image = Image.open(img)
    mask = Image.new("L", image.size, 0)
    for pol in polygons:
        if len(pol) >= 2:
            ImageDraw.Draw(mask).polygon(pol, fill="white", outline=outline, width=edge_width)

    return mask


def get_fruits_size_info(polygons, img_path):
    """

      Calculer les ratio minor_axis/major_axis pour chaque fruit

    """
    mask_image = pols_to_mask(polygons, img_path)
    mask_image = np.array(mask_image)
    ratios = get_min_major_axis_ratio(mask_image)

    return ratios


# Fonction pour visualiser les axes majeur et mineur sur une image donnée
def get_min_major_axis_ratio(mask_image):
    """
    Calcul le ratio entre le minor_axis / major_axis
    """
    ratios = []
    labels = label(mask_image)
    regions = regionprops(labels)
    for props in regions:
        major_axis = props.axis_major_length  # longueur du fruit
        minor_axis = props.axis_minor_length  # largeur du fruit
        if major_axis > 0 and minor_axis > 0:
            ratio = minor_axis / (major_axis + 0.000001)  # Ratio entre la largeur et la longueur
            ratios.append(ratio)

    return ratios


def get_correct_masks(masque_cadre, boites=None, masques=None):
    """
    Ne retenir que les masques de fruits qui se trouve dans la zone du cadre
    """
    mask_array = None
    if isinstance(masque_cadre, str):
        mask_array = io.imread(masque_cadre)
    else:
        mask_array = masque_cadre
    masques_retenus = []
    # boxes_centers = yolo_pred_to_points(boites)
    # boxes_centers = [(boxes_centers[i], boxes_centers[i+1]) for i in range(0, len(boxes_centers), 2)]
    H, W = mask_array.shape
    for bbox_center, msk in zip(boites[:], masques):
        col, row = bbox_center
        assert col < W, f"col {col} is out of bounds. Max width is {W}"
        assert row < H, f"row {row} is out of bounds. Max height is {H}"
        if mask_array[row, col] != 0:
            masques_retenus.append([tuple(e) for e in msk])

    return masques_retenus


def check_and_compute_weight(ratios):
    """Catégorie des fruits selon la moyenne du ratio
      A : <= 0.55 -> Fusiforme : 11.34 g
      B : 0.55 < largeur/longueur < 0,7 -> Ovale, ovale apiculée et goutte : 10.3 g
      C : >= 0.7 : Arrondie et globuleux : 6.29 g

      return:
        Type_fruit, poids du fruit
    """
    if len(ratios) > 0:
        mean_ratio = np.mean(ratios)
        if mean_ratio <= 0.55:
            return "Fusiforme", 11.34 * len(ratios)
        elif mean_ratio < 0.7:
            return "Ovale", 10.3 * len(ratios)
        else:
            return "Arrondie", 6.29 * len(ratios)
    else:
        return "Aucun", 0 * len(ratios)


def clip_boxes(list_boxes, image_shape):
    """
    Clip the predicted boxes to keep them in the range
    """
    H, W = image_shape
    final_boxes = []
    for bbox in list_boxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin = max(0, x1)
        ymin = max(0, y1)
        xmax = min(x2, W)
        ymax = min(y2, H)
        final_boxes.append([xmin, ymin, xmax, ymax])

    return final_boxes


def pols_list_to_tuple(polygones):
    p_final = []
    for p in polygones:
        p_final.append([tuple(e) for e in p])
    return p_final


def show_pols(polygons, pil_image, transparence=100):
    """
    Afficher les polygones sur l'image
    """
    for pol in polygons:
        color = (random.randint(50, 255), random.randint(75, 255), random.randint(100, 255), transparence)
        ImageDraw.Draw(pil_image, "RGBA").polygon(pol, fill=color, outline=color)
    return pil_image


def pols_to_mask(polygons, img_path, edge_width=3, fill_color="white", outline="black"):
    """
    polygons : [[(x, y), (x, y), ...]]
    Créer un mask des fruits à partir des polygones
    """
    image = Image.open(img_path)
    mask = Image.new("L", image.size, 0)
    for pol in polygons:
        ImageDraw.Draw(mask).polygon(pol, fill=fill_color, outline=outline, width=edge_width)

    return mask


# Fonction pour visualiser les axes majeur et mineur sur une image donnée
def show_axes_on_fruits(image, mask_image, width=2):
    """
    Afficher les demis axes mineurs et majeur sur les fruits
    """
    labels = label(mask_image)
    regions = regionprops(labels)
    pil_msk_img = image
    for props in regions:
        y0, x0 = props.centroid

        orientation = props.orientation
        # Major axis
        x2s_left = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2s_left = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
        x2e_right = x0 + math.sin(orientation) * 0.5 * props.axis_major_length
        y2e_right = y0 + math.cos(orientation) * 0.5 * props.axis_major_length

        # Minor axis
        x1e_right = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1e_right = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x1s_left = x0 - math.cos(orientation) * 0.5 * props.axis_minor_length
        y1s_left = y0 + math.sin(orientation) * 0.5 * props.axis_minor_length

        major_axis_coors = [x2s_left, y2s_left, x2e_right, y2e_right]
        minor_axis_coors = [x1s_left, y1s_left, x1e_right, y1e_right]

        ImageDraw.Draw(pil_msk_img).line(major_axis_coors, fill="red", width=width)
        ImageDraw.Draw(pil_msk_img).line(minor_axis_coors, fill="blue", width=width)

    return pil_msk_img


class ModelSegmentationFruits(object):
    def __init__(self, checkpoint):
        """
        Modèle de segmentation des fruits. Il prend un ckeckpoint pour initialiser le modèle YOLO
        Lien de la documentation YOLO: https://docs.ultralytics.com/fr
        :param checkpoint:
        """
        self.checkpoint = checkpoint
        self.model = YOLO(self.checkpoint)

    def train(self, data_yaml_path, save_path, save_path_subfolder, augmentations=None,
              epochs=100, resume=False, imgsz=640, patience=100, batch=16, fraction=1., **kwargs):
        """
        data_yaml_path: chemin vers le fichier data.yaml
        save_path: chemin où les résultats seront sauvegardés
        save_path_subfolder: sous dossier dans lequel les résultats seront sauvegardés
        augmentations: les augmentations à appliquer (voir documentation de YOLOv8 dans ultralytics)
        epochs: Nombre d'époques à entrainer le modèle
        resume: Reprendre l'entrainement à partir de l'entrainement précédent ou non
        imgsz : taille de l'image d'entrainement. nombre entier ou tuple (width, height)
        patience: nombre d'époque à attentre sans amélioration avant d'interrompre l'entrainement
        batch: la taille du batch
        """
        if augmentations is None:
            # Augmentations à appliquer
            augmentations = {
                "hsv_h": 0., "hsv_s": 0.,
                "hsv_v": 0., "degrees": 0.,
                "translate": 0., "perspective": 0.0,
                "shear": 0.0, "scale": 0.,
                "flipud": 0.0, "fliplr": 0.5,
                "mosaic": 0, "close_mosaic": 0
            }

        if resume:
            self.model = YOLO(os.path.join(save_path, save_path_subfolder, "weights", "last.pt"))
            epochs = epochs - pd.read_csv(os.path.join(save_path, save_path_subfolder, "results.csv")).shape[0]

        # Entrainer le modèle
        self.model.train(data=data_yaml_path, epochs=epochs, resume=resume,
                         name=save_path_subfolder, project=save_path,
                         imgsz=imgsz, patience=patience, optimizer="Adam",
                         batch=batch, **augmentations, fraction=fraction, **kwargs
                         )

    def val(self, data_yaml_path, split="val", iou=0.7, conf=0.5, imgsz=960, batch=8, max_det=200, **kwargs):
        """
        Validation du modèle entrainé
        """
        results = self.model.val(
            data=data_yaml_path,
            split=split,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            batch=batch,
            max_det=max_det,
            **kwargs
        )[0]

        return results

    def predict(self, source, imgsz=(960, 960), conf=0.25, iou=0.7, max_det=200, **kwargs):
        """
        :param max_det: Nombre maximal de détections
        :param iou: valeur minimal d'iou pour le Non Max Suppression
        :param conf: Probabilité minimal à considérer
        :param source: Image à utiliser pour la prédiction
        :param imgsz: taille de l'image pour la prédiction
        :param kwargs: arguments à passer au modèle YOLO. Voir la documentation d'ultralytics
        https://docs.ultralytics.com/modes/predict/#inference-arguments
        source, conf, iou
        :return: resultats
        """
        resultats = self.model.predict(source=source, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, **kwargs)
        return resultats


# -------
def get_yolo_pred(model, image_path, img_size=None, conf=0.5, iou=0.7, masque_du_cadre=None):
    """
    Prédire et retourner les centres des boites ainsi que les masques de YOLO
    return nombre_predit, list_ratios, liste_polygones_fruits
    """
    pred_size = img_size
    W, H = Image.open(image_path).size
    if not isinstance(img_size, int):
        pred_size = (H, W)

    res = model.model(source=image_path, conf=conf, iou=iou, imgsz=pred_size, verbose=False, max_det=240)
    # boites = res[0].boxes.xyxy.cpu().numpy().tolist()
    boites = res[0].boxes.xywhn.cpu()

    boites[:, 0] *= W
    boites[:, 1] *= H
    boites = boites.numpy().tolist()
    boites_centers = [(int(e[0]), int(e[1])) for e in boites]
    if len(boites_centers) > 0:
        masques = [e.tolist() for e in res[0].masks.xyn]
        masques_finaux = []
        for pol in masques:
            pl = []
            for e in pol:
                pl.append([e[0] * W, e[1] * H])
                masques_finaux.append(pl)

        final_pred_msks = pols_list_to_tuple(masques_finaux)
        if masque_du_cadre is not None:
            # Retenir que les masques qui sont dans le cadre
            final_pred_msks = get_correct_masks(masque_du_cadre, boites=boites_centers, masques=masques_finaux)

        # Calculer les ratio largeur/longueur des fruits retenus et la moyenne pour l'image
        ratios_minor_major_axis = get_fruits_size_info(final_pred_msks, image_path)

        nombre_predit_ = len(final_pred_msks)
        cat, poids = check_and_compute_weight(ratios_minor_major_axis)

        return nombre_predit_, cat, poids, final_pred_msks, ratios_minor_major_axis

    else:
        return 0, "Aucun", 0, [], []


def get_true_number_fruit_type_weight(label_path, image_path, mask_p=None, save_fold=None):
    """
    Récupérer le nombre réel de fruits, la catégorie et le poids réel
    :param label_path:
    :param image_path:
    :param mask_p:
    :param save_fold:
    :return:
    """
    # Nombre, dimension des fruits annotées
    masques_annotees = get_polygons(path_to_label_file=label_path, image_path=image_path)
    boites_centers_annotees = polygons_to_points(masques_annotees)
    correct_true_masks = pols_list_to_tuple(masques_annotees)
    if mask_p is not None:
        correct_true_masks = get_correct_masks(mask_p, boites=boites_centers_annotees, masques=masques_annotees)
    ratios_reels = get_fruits_size_info(correct_true_masks, image_path)
    nombre_reel = len(ratios_reels)
    categorie_reel, poids_reel = check_and_compute_weight(ratios_reels)  # Poids en gramme
    if save_fold is not None:
        # Sauvegarder les ratios
        bare_name_ = image_path.split("/")[-1].split(".")[0] + "True_label_ratio" + ".txt"
        save_content(ratios_reels, path=os.path.join(save_fold, bare_name_))

    return nombre_reel, categorie_reel, poids_reel


def get_tilled_yolo_pred(model_path, img_path, iou=0.7, conf=0.3, slice_size=480, overlap_pixels=200,
                         pred_size=640, mask_path=None, save_fold=None):
    """ Une prédiction en tuilage qui découpera l'image en plusieurs morceaux
    Prédire et retourner les centres des boites ainsi que les masques de YOLO
    return nombre_predit, list_ratios, liste_polygones_fruits
    """
    overlap = (overlap_pixels / slice_size) * 100
    img = cv2.imread(img_path)
    element_crops = MakeCropsDetectThem(
        image=img,
        model_path=model_path,
        segment=True,
        imgsz=pred_size,
        overlap_x=overlap,  # 50% overlap
        overlap_y=overlap,  # 50% overlap
        conf=conf,
        iou=iou,
        shape_x=slice_size,  # Crop width
        shape_y=slice_size,  # Crop height
        resize_initial_size=True,  # Resize prediction to the original image size
        show_crops=False
    )

    result = CombineDetections(
        element_crops,
        nms_threshold=iou,  # IoU/IoS threshold for suppression
        match_metric='IOU'
    )

    pred_boxes = result.filtered_boxes
    pred_msks = result.filtered_masks

    pols = []
    if len(pred_boxes) > 0:
        for msk in pred_msks:
            mask_contours, _ = cv2.findContours(
                msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(mask_contours) > 0:
                pol = mask_contours[0].reshape((mask_contours[0].shape[0], -1)).tolist()
                pol = [tuple(e) for e in pol]
                pols.append(pol)

            image_shape = Image.open(mask_path).size  # -> (width, height)
            pred_boxes_correct = clip_boxes(pred_boxes, (image_shape[1], image_shape[0]))
            boites_centers = bboxes_to_points(pred_boxes_correct)

        if len(boites_centers) > 0:
            final_pred_msks = pols
            if mask_path is not None:
                # Retenir que les masques qui sont dans le cadre
                final_pred_msks = get_correct_masks(mask_path, boites=boites_centers, masques=pols)

            # Calculer les ratio largeur/longueur des fruits retenus et la moyenne pour l'image
            ratios_minor_major_axis = get_fruits_size_info(final_pred_msks, img_path)
            # Sauvegarder les ratios
            if save_fold is not None:
                bare_name = img_path.split("/")[-1].split(".")[0] + ".txt"
                save_content(ratios_minor_major_axis, path=os.path.join(save_fold, bare_name))
            nombre_predit_ = len(final_pred_msks)
            categorie_predit_, poids_predit_ = check_and_compute_weight(ratios_minor_major_axis)
            return nombre_predit_, categorie_predit_, poids_predit_, final_pred_msks, ratios_minor_major_axis
        else:
            return 0, "Aucun", 0, [], []
    else:
        return 0, "Aucun", 0, [], []


def predictions_for_validation(model_checkpoint, predicted_masks_path, true_masks_path, images_path, true_labels_path,
                               save_folder_path, file_name, slice_prediction=False, confs=[0.3, 0.4], slice_sizes=None,
                               overlap_pixels=None, image_size=None, iou=0.7):
    """
        Créer un fichier de validation selon différents niveaux de probabilité. Il faudrait spécifier le chemin vers le
        modèle préentrainé ayant l'extension .pt, les dossiers contenant les masques des cadres prédits, les vrais
        masques des cadres, les images, les fruits annotées et le dossier dans lequel on veut sauvegarder le fichier
        .csv qui sera créé
    """

    noms_images = []
    confs = []
    nombres_predits = []
    poids_predits = []
    categories_predits = []
    nombres_reels = []
    poids_reels = []
    categories_reels = []
    overlap_pix = []
    slc_size = []

    model = ModelSegmentationFruits(model_checkpoint)
    pred_masks = sorted(glob.glob(os.path.join(predicted_masks_path, "*.png")))
    true_masks = sorted(glob.glob(os.path.join(true_masks_path, "*.png")))
    images_p = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
    labels_p = sorted(glob.glob(os.path.join(true_labels_path, "*.txt")))
    save_folder = save_folder_path
    file_name = file_name

    def do_slice_predictions(**kwargs):
        """
            Prédiction en tuilage. Elle prend en paramètre le checkpoint du modèle, les dossiers contenant les masques
            des cadres prédits, les vrais masques des cadres, les images, les fruits annotées et le dossier dans lequel
            on veut sauvegarder le fichier .csv qui sera créé, les confs, la taille du slice et le nombre de pixels
            de superpositions (overlap_pixels)

            kwargs : model_path, pred_masks, true_masks, images_p, labels_p, confs, slice_sizes, overlap_pixels, iou
        """

        for pred_msk_path, true_msk_path, img_p, lbl_p in tqdm.tqdm(
                zip(kwargs["pred_masks"], kwargs["true_masks"], kwargs["images_p"], kwargs["labels_p"]),
                total=len(labels_p)
        ):
            nom_img = img_p.split("/")[-1]

            # Nombre de fruits du cadre, catégorie réel et poids réel
            nombre_reel, categorie_reel, poids_reel = get_true_number_fruit_type_weight(label_path=lbl_p,
                                                                                        image_path=img_p,
                                                                                        mask_p=true_msk_path)

            for conf in kwargs["confs"]:
                for slice_size in kwargs["slice_sizes"]:
                    for overlap_pixels in kwargs["overlap_pixels"]:
                        # Prédire en utilisant YOLO
                        nombre_predit, categorie_predit, poids_predit = get_tilled_yolo_pred(
                            kwargs["model_path"],
                            img_p,
                            iou=kwargs["iou"],
                            conf=conf,
                            slice_size=slice_size,
                            overlap_pixels=overlap_pixels,
                            pred_size=slice_size,
                            mask_path=pred_msk_path
                        )

                        noms_images.append(nom_img)
                        nombres_predits.append(nombre_predit)
                        poids_predits.append(poids_predit)
                        categories_predits.append(categorie_predit)
                        nombres_reels.append(nombre_reel)
                        poids_reels.append(poids_reel)
                        categories_reels.append(categorie_reel)
                        confs.append(conf)
                        slc_size.append(slice_size)
                        overlap_pix.append(overlap_pixels)

    def do_no_slice_predictions(**kwargs):
        """
        Effectuer les prédictions sans tuilage pour la validation.
        kwargs : model_path, pred_masks, true_masks, images_p, labels_p, confs, iou, image_size
        """

        model = ModelSegmentationFruits(kwargs["model_path"])

        for pred_msk_path, true_msk_path, img_p, lbl_p in tqdm.tqdm(
                zip(kwargs["pred_masks"], kwargs["true_masks"], kwargs["images_p"], kwargs["labels_p"]),
                total=len(labels_p)
        ):
            nom_img = img_p.split("/")[-1]

            # Nombre de fruits du cadre, catégorie réel et poids réel
            nombre_reel, categorie_reel, poids_reel = get_true_number_fruit_type_weight(label_path=lbl_p,
                                                                                        image_path=img_p,
                                                                                        mask_p=true_msk_path)

            for conf in kwargs["confs"]:
                # Prédire en utilisant YOLO
                nombre_predit, categorie_predit, poids_predit = get_yolo_pred(
                    model,
                    img_p,
                    img_size=kwargs["image_size"],
                    conf=conf,
                    iou=kwargs["iou"],
                    masque_du_cadre=pred_msk_path)

                noms_images.append(nom_img)
                nombres_predits.append(nombre_predit)
                poids_predits.append(poids_predit)
                categories_predits.append(categorie_predit)
                nombres_reels.append(nombre_reel)
                poids_reels.append(poids_reel)
                categories_reels.append(categorie_reel)
                confs.append(conf)

    if slice_prediction:
        do_slice_predictions(model_path=model_checkpoint, pred_masks=pred_masks, true_masks=true_masks,
                             images_p=images_p, labels_p=labels_p, confs=confs, slice_sizes=slice_sizes,
                             overlap_pixels=overlap_pixels, iou=iou)
    else:
        do_no_slice_predictions(
            model_path=model_checkpoint, pred_masks=pred_masks, true_masks=true_masks,
            images_p=images_p, labels_p=labels_p, confs=confs, iou=iou, image_size=image_size
        )

    # Create and save the .csv file
    data = {
        "image": noms_images,
        "conf": confs,
        "n_predit": nombres_predits,
        "poids_predit": poids_predits,
        "categorie_predit": categories_predits,
        "n_reel": nombres_reels,
        "poids_reel": poids_reels,
        "categorie_reel": categories_reels
    }

    if slice_prediction:
        data["overlap"] = overlap_pix,
        data["slice_size"] = slc_size

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_folder, file_name), index=False)


def get_trees_ids(dataframe):
    """
    Récupérer la liste des identifiant des arbres
    """
    return dataframe.arbreID.unique().tolist()


def get_tree_images(tree_id, dataframe, tree_id_field="arbreID", image_field="Image"):
    """
    Récupère les 4 images correspondant à l'arbre donné
    tree_id : identifiant de l'arbre
    dataframe : dataframe pandas contenant les données

    return:
      list de 4 images
    """
    donnees = dataframe[dataframe[tree_id_field] == tree_id]
    images = donnees[image_field].to_list()
    return images


def get_predicted_mask(image_path, mask_model_checkpoint, imgs=960, post_process=True, device=get_device()):
    """
    Prédire le cadre se trouvant dans l'image
    """
    model_cadre = SegmentationModel(mask_model_checkpoint)
    msk_cadre_predit = model_cadre.predict(image=image_path, image_size=imgs, post_process=post_process, device=device)
    msk_cadre = msk_cadre_predit * 255

    return msk_cadre


def draw_n_fruits_on_image(image, text, position=(20, 20), rectangle_fill_color=(255, 255, 255, 180),
                           text_color="black", text_size=25):
    """
    Afficher les fruits sur les images y compris du texte à une position donnée
    :param image:
    :param text:
    :param position:
    :param rectangle_fill_color:
    :param text_color:
    :param text_size:
    :return:
    """
    font_path = Path(__file__).parent.resolve(strict=True).parent / "configs" / "ARIAL.TTF"
    font = ImageFont.truetype(font_path, text_size)

    image = image
    if isinstance(image, str):
        image = Image.open(image)
    draw = ImageDraw.Draw(image, "RGBA")

    left, top, right, bottom = draw.textbbox(position, text, font=font)
    draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=rectangle_fill_color)
    draw.text(position, text, font=font, fill=text_color)
    return image


def predict_fruits_yield_by_tree(images_folder, fruits_model_checkpoint, cadre_model_checkpoint, save_path, filename,
                                 predict_mask_size=960, post_process_mask=True, slice_prediction=False,
                                 slice_size=640, overlap_pixels=200, iou=0.7, conf=0.3,
                                 fruits_no_slice_prediction_size=None, save_predictions_images=False):
    """
    Prédire le rendement en fruits présent sur les images pour chaque arbre
    :return:
    """
    df_images = get_images_dataframe(images_folder)
    trees_ids = get_trees_ids(df_images)

    dict_donnes_arbres = {
        "tree_id": [],
        "categorie_fruit": [],
        "nombre_moyen_fruits": [],
        "poids_du_fruit": [],
        "poids_moyen_cadre": [],
        "nombre_photos_de_l_arbre": []
    }

    for tree_id in tqdm.tqdm(trees_ids, total=len(trees_ids)):
        liste_images = get_tree_images(tree_id, df_images)
        list_imgs_paths = [os.path.join(images_folder, img_name) for img_name in liste_images]

        nombres_fruits = []
        noms_images = []
        liste_ratios = []
        images_annotees = []

        model_fruits = ModelSegmentationFruits(checkpoint=fruits_model_checkpoint)

        for img_p in list_imgs_paths:
            img_p += ".jpg"
            nom_image = Path(img_p).name
            noms_images.append(nom_image)

            # ------- Pour chaque image : prédire le cadre et les fruits ------------
            # Prédire le cadre
            mask_predit = get_predicted_mask(image_path=img_p, mask_model_checkpoint=cadre_model_checkpoint,
                                             imgs=predict_mask_size, post_process=post_process_mask)

            # Prédire les fruits : tuillage ou non
            nbr_fruits, ratios, masques = None, None, None
            if not slice_prediction:
                nbr_fruits, _, _, masques, ratios = get_yolo_pred(model_fruits, img_p, img_size=fruits_no_slice_prediction_size,
                                                            conf=conf, iou=iou, masque_du_cadre=mask_predit)
            else:
                nbr_fruits, _, _, masques, ratios = get_tilled_yolo_pred(model_path=fruits_model_checkpoint,
                                    img_path=img_p, iou=iou, conf=conf, slice_size=slice_size,
                                    overlap_pixels=overlap_pixels, pred_size=slice_size, mask_path=mask_predit)

            nombres_fruits.append(nbr_fruits)

            # Ajouter le ratio à la liste des ratios
            liste_ratios += ratios

            if save_predictions_images:
                # Annoté l'image avec les polygones prédits
                image_annotee = show_pols(masques, Image.open(img_p), transparence=130)
                images_annotees.append(image_annotee)

        # Déterminer la catégorie du fruit et le poids unitaire
        categorie, poids_fruit = check_and_compute_weight(liste_ratios)

        if save_predictions_images:
            # Annotation finale et sauvegarde des images
            for img_name, n_fruits, img in zip(noms_images, nombres_fruits, images_annotees):
                final_img_annotated = draw_n_fruits_on_image(
                    image=img, text=f"Nombre de fruits: {n_fruits} | Catégorie: {categorie} | Poids unitaire : {poids_fruit} g | Poids total : {'{:.2f}'.format(n_fruits * poids_fruit)} g"
                )
                final_img_annotated.save(Path(save_path) / (str(img_name)))

        dict_donnes_arbres["tree_id"].append(tree_id)
        dict_donnes_arbres["categorie_fruit"].append(categorie)
        dict_donnes_arbres["nombre_moyen_fruits"].append(math.ceil(sum(nombres_fruits) / len(nombres_fruits)))
        dict_donnes_arbres["poids_du_fruit"].append(poids_fruit)
        dict_donnes_arbres["poids_moyen_cadre"].append(math.ceil(sum(nombres_fruits) / len(nombres_fruits)) * poids_fruit)
        dict_donnes_arbres["nombre_photos_de_l_arbre"].append(len(liste_images))

        pd.DataFrame(dict_donnes_arbres).to_csv(os.path.join(save_path, filename), index=False)


if "__name__" == "__main__":
    pass

