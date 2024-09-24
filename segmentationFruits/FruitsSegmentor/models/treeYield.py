# Import packagees
from pathlib import Path

import geopandas as gpd
import pandas as pd

pd.options.mode.use_inf_as_na = True
import numpy as np
import math

import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

import os, glob
from skimage import io

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_theme(context='paper', style='whitegrid', font_scale=2)  # darkgrid

import math
import pickle


def get_trees_shape_and_yield(dataframe, diameter_column, height_column, yield_column=None):
    """
    Récupérer le diamètre, la hauteur de la cîme et le rendement aussi s'il existe
    :param diameter_column:
    :param height_column:
    :param yield_column:
    :return:
    """
    cols = [diameter_column, height_column, yield_column] if yield_column is not None else [diameter_column,
                                                                                            height_column]
    try:
        columns_values = dataframe[cols]
        return columns_values
    except:
        print("Assurer vous que les noms des colonnes sont exactes !")


def get_dataframe_columns_values(dataframe, liste_columns):
    """
    Récupère les colonnes du dataframe
    :param dataframe:
    :param liste_columns:
    :return:
    """
    df_cols = dataframe.columns.values.tolist()
    for e in liste_columns:
        if e not in df_cols:
            raise ValueError(f"La colonne {e} ne se trouve pas dans la liste des colonnes du dataframe")

    return dataframe[liste_columns]


def get_dataframe(file_path, columns=None):
    """
    Récupérer un dataframe donné en utilisant le chemin de fichier. Possible de spécifier les colonnes voulues
    :param columns:
    :param file_path: liste des colonnes à retourner
    :return:
    """
    df = pd.read_csv(file_path)
    if columns is not None:
        return get_dataframe_columns_values(df, columns)

    return df


def get_geodataframe(file_path, columns=None):
    """
    Récupérer un dataframe donné en utilisant le chemin de fichier. Possible de spécifier les colonnes voulues
    :param columns:
    :param file_path: liste des colonnes à retourner
    :return:
    """
    df = gpd.read_file(file_path, ignore_geometry=True)
    if columns is not None:
        return get_dataframe_columns_values(df, columns)

    return df


def analyse_trees_canopy_shape(dataframe, canopy_diameter_col, canopy_height_col):
    """
    Identifier le nombre d'arbres avec diamètre/hauteur du houppier ou les deux
    :param dataframe:
    :param canopy_diameter_col:
    :param canopy_height_col:
    :return:
    """
    n_diam = dataframe[dataframe[canopy_diameter_col].notnull()].shape[0]
    print("Nombre d'arbres dont le diamètre de la cime a été renseigné :", n_diam)

    n_haut = dataframe[dataframe[canopy_height_col].notnull()].shape[0]
    print("Nombre d'arbres dont la hauteur de la cime a été renseigné :", n_haut)


def val_to_float(e):
    """
    Transforme des nombres en texte au format float avec comme séparateur décimal le .
    :param e:
    :return:
    """
    a = str(e).replace(",", ".").replace("'", ".")
    return float(a)


def df_cols_to_float(dataframe, cols):
    """
    Convertir les colonnes du dataframe en flot
    :param dataframe:
    :param cols:
    :return:
    """
    for col in cols:
        dataframe[col] = dataframe[col].map(val_to_float)

    return dataframe


def predict_tree_yield_without_ajustment(trees_shp_path=None, tree_id_col_shp="ech_id",
                                         trees_yield_from_images_csv=None,
                                         tree_id_col_images="arbreID", diam_col="ech_diam_c", height_col="ech_h_cime",
                                         col_densite_fruits="nombre_moyen_fruits", col_poids_fruits="poids_du_fruit"):
    """
    Prédire le rendement des arbres en supposant que la forme du houppier est un demi ellipsoide
    :param save_file_path:
    :param model_ajustement_path:
    :param col_rendement_ajuste:
    :param model_ajustement:
    :param col_poids_fruits:
    :param col_densite_fruits:
    :param trees_shp_keep_cols: colonnes à garder : une liste [cola, colob, etc.]
    :param tree_id_col_images:
    :param tree_id_col_shp:
    :param trees_shp_path:
    :param trees_yield_from_images_csv:
    :param diam_col:
    :param height_col:
    :param yield_col:
    :return:
    """
    # Vérifier sur les fichiers existent d'abord
    assert Path(trees_shp_path).exists(), f"{trees_shp_path} introuvable"
    assert Path(trees_yield_from_images_csv).exists(), f"{trees_yield_from_images_csv} introuvable"

    # Lecture des dataframes
    # df_trees = None
    if Path(trees_shp_path).suffix == ".csv":
        df_trees = get_dataframe(trees_shp_path)
    else:
        df_trees = get_geodataframe(file_path=trees_shp_path)

    # Supprimer les arbres dont certaines valeurs de hauteur ou diamètre sont na
    df_trees = df_trees[df_trees[diam_col].notna()]
    df_trees = df_trees[df_trees[height_col].notna()]

    df_rend_images = pd.read_csv(trees_yield_from_images_csv)

    # Arbres dont la hauteur et le diamètre du houppier ne sont pas nul
    arbres_h_diam_not_nul = [e[0] for e in df_trees[[tree_id_col_shp, diam_col, height_col]].values.tolist() if
                             e[2] is not None and e[1] is not None]
    data = df_trees.query(f"{tree_id_col_shp} in {arbres_h_diam_not_nul}")
    data[height_col] = data[height_col].map(val_to_float)
    data[diam_col] = data[diam_col].map(val_to_float)

    # Fusionner les deux dataframe: avoir les info des arbres et leurs rendements sur les images
    data_rendement = pd.merge(data, df_rend_images, how="left", left_on=tree_id_col_shp,
                              right_on=tree_id_col_images)

    # Calcul de la surface du houppier assimilé à un demi ellipsoide
    p = 1.6075
    a_p = c_p = (data_rendement[diam_col] / 2) ** p
    b_p = (data_rendement[height_col] ** p)
    data_rendement["surface_cime_demi_ellipsoide"] = 2 * math.pi * ((a_p * b_p + a_p * c_p + b_p * c_p) / 3) ** (1 / p)

    # Calcul du nombre total de fruits et du poids des fruits pour l'ensemble du houppier
    data_rendement["nombre_fruits_houppier"] = (data_rendement["surface_cime_demi_ellipsoide"] * data_rendement[
        col_densite_fruits]) / 0.09
    data_rendement["rend_predit_houppier"] = (data_rendement["nombre_fruits_houppier"] * data_rendement[
        col_poids_fruits]) / 1000

    data_rendement = data_rendement[data_rendement["rend_predit_houppier"].notna()]

    return data_rendement


def predict_tree_yield(trees_shp_path=None, tree_id_col_shp="ech_id", trees_yield_from_images_csv=None,
                       tree_id_col_images="arbreID", diam_col="ech_diam_c", height_col="ech_h_cime",
                       col_densite_fruits="nombre_moyen_fruits", col_poids_fruits="poids_du_fruit",
                       col_rendement_ajuste="rend_20xy", model_ajustement_path=None, save_file_path="resultats.csv"):
    """
    Prédire le rendement des arbres en supposant que la forme du houppier est un demi ellipsoide
    :param save_file_path:
    :param model_ajustement_path:
    :param col_rendement_ajuste:
    :param model_ajustement:
    :param col_poids_fruits:
    :param col_densite_fruits:
    :param trees_shp_keep_cols: colonnes à garder : une liste [cola, colob, etc.]
    :param tree_id_col_images:
    :param tree_id_col_shp:
    :param trees_shp_path:
    :param trees_yield_from_images_csv:
    :param diam_col:
    :param height_col:
    :param yield_col:
    :return:
    """
    data_rendement = predict_tree_yield_without_ajustment(trees_shp_path=trees_shp_path,
                                                          tree_id_col_shp=tree_id_col_shp,
                                                          trees_yield_from_images_csv=trees_yield_from_images_csv,
                                                          tree_id_col_images=tree_id_col_images,
                                                          diam_col=diam_col, height_col=height_col,
                                                          col_densite_fruits=col_densite_fruits,
                                                          col_poids_fruits=col_poids_fruits)

    if model_ajustement_path is not None:
        # Ajuster le rendement en fruits en utilisant le modèle d'ajustement
        rend_predit = data_rendement["rend_predit_houppier"].values[..., np.newaxis]
        assert Path(model_ajustement_path).exists(), f"{model_ajustement_path} non trouvé. Corrigez le chemin !"
        with open(model_ajustement_path, "rb") as f:
            model_ajustement = pickle.load(f)
            rend_ajuste = model_ajustement.predict(rend_predit)
            # Ajouter la colonne du rendement ajusté
            data_rendement[col_rendement_ajuste] = rend_ajuste

    # Enregistrer les résultats de prédiction
    data_rendement.to_csv(save_file_path, index=False)


def linear_regression(x_train_data, y_train_data, x_test_data, y_test_data):
    """
        Effectuer une regression linéaire et retourner les résultats
    """
    reg = LinearRegression()
    reg.fit(x_train_data, y_train_data)
    y_predit = reg.predict(x_test_data)
    mae = mean_absolute_error(y_test_data, y_predit)
    r_carre = '{:.2f}'.format(reg.score(x_test_data, y_test_data))
    a = '{:.2f}'.format(reg.coef_[0])
    b = '{:.2f}'.format(reg.intercept_)

    return {
        "coefs_a_b": [a, b],
        "mae": "{:.2f}".format(mae),
        "r2": r_carre,
        "regressor": reg
    }


def regress_and_visualize(x_, y_, xtrain, ytrain, xtest, ytest,
                          xlabel="Rendement modélisé (Kg)",
                          ylabel="Rendement réel (Kg)",
                          title=None, save_path=None,
                          save_model_path="model_ajustement.pkl"
                          ):
    """
        Effectuer la régression linéaire et représenter les données
    """
    import pickle

    # Definition du mmodel
    res = linear_regression(
        x_train_data=xtrain, y_train_data=ytrain,
        x_test_data=xtest, y_test_data=ytest)
    a, b = res["coefs_a_b"]
    mae = res["mae"]
    r2 = res["r2"]
    model_regression = res["regressor"]
    rend_ajuste = model_regression.predict(x_)

    # Save the model
    if save_model_path is not None:
        with open(save_model_path, 'wb') as f:
            pickle.dump(model_regression, f)

    # Prédire sur l'ensemble des données
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_, rend_ajuste, label=f"y = {a}x + {b} (R²={r2} - mae={mae})")
    ax.scatter(x=x_, y=y_, alpha=0.3, color="blue")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    plt.show()


def create_adjustment_model(trees_shp_path=None, tree_id_col_shp="ech_id", trees_yield_from_images_csv=None,
                            tree_id_col_images="arbreID", diam_col="ech_diam_c", height_col="ech_h_cime",
                            col_densite_fruits="nombre_moyen_fruits", col_poids_fruits="poids_du_fruit",
                            col_rendement_reel="ech_ren_22", random_state=1, train_size=0.7, save_path="None"):
    """

    :return: None
    """
    # Prédire le rendement sur l'ensemble du houppier sans ajustement
    data_rendement = predict_tree_yield_without_ajustment(trees_shp_path=trees_shp_path,
                                                          tree_id_col_shp=tree_id_col_shp,
                                                          trees_yield_from_images_csv=trees_yield_from_images_csv,
                                                          tree_id_col_images=tree_id_col_images,
                                                          diam_col=diam_col, height_col=height_col,
                                                          col_densite_fruits=col_densite_fruits,
                                                          col_poids_fruits=col_poids_fruits)

    # Supprimer les valeurs de rendement réel dont la valeur est nulle ou n'est pas renseignée
    def correct_virgules(e):
        return str(e).replace(",", ".").replace("'", ".")

    y = data_rendement[col_rendement_reel].map(correct_virgules)
    y, x = y[y.notna()], data_rendement["rend_predit_houppier"][y.notna()]

    def is_number(e):
        """Return True if the string is a float otherwise False"""
        return e.isdecimal()
    true_indices = y.map(is_number)

    # Récupération des colonnes
    y = y[true_indices].map(val_to_float)
    x = x[true_indices].values[..., np.newaxis]

    # Répartir les données
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=train_size, random_state=random_state)

    # Crééer le modèle d'ajustement et le sauvegarder
    regress_and_visualize(x_=x, y_=y, xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest,
                          xlabel="Rendement modélisé (Kg)",
                          ylabel="Rendement réel (Kg)",
                          title=None, save_path=save_path,
                          save_model_path=str(Path(save_path) / "model_ajustement.pkl")
                          )