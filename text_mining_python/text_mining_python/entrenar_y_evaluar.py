# -*- coding: utf-8 -*-
"""
Este script lee el resultado del script "de_html_a_tabla.py" como dataset de entrenamiento y validacion. Entrena n clasificadores y los evalua con cross-validation.
"""
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, auc
from sklearn.multiclass import OneVsRestClassifier
import joblib
from sklearn.feature_selection import SelectKBest, chi2
from typing import Dict
import numpy as np

# nombres de los archivos a leer
VECTORS_FILE = "vectores.joblib"
TARGETS_FILE = "targets.joblib"
FEATURE_NAMES_FILE = "features.joblib"

def _calcular_auc_por_clase(targets_reales:np.ndarray, targets_preds:np.ndarray) -> Dict[int, float]:
    """
    Computa la curva ROC y AUC para cada clase.
    :param targets_reales: Un vector de targets reales representados en 1-hot encoding.
    :param targets_preds: Un vector de targets predichos representados en 1-hot encoding.
    :return: Un diccionario de indice de categoria -> AUC de esa categoria
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_clases = targets_preds.shape[1]
    for i in range(n_clases):
        fpr[i], tpr[i], _ = roc_curve(targets_reales[:, i], targets_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc


def calcular_e_imprimir_auc(clasificador, train_fold_selected, train_targets_binarios_por_clase, test_fold_selected, test_targets_binarios_por_clase):
    """
    Calcular e imprime el AUC para cada categoria, utilizando el clasificador y los folds de entrenamiento y test.
    :param clasificador: Un clasificador de scikit-learn.
    :param train_fold_selected: Fold de entrenamiento
    :param train_targets_binarios_por_clase: Categorias del fold de entrenamiento, en 1-hot encoding.
    :param test_fold_selected: Fold de test
    :param test_targets_binarios_por_clase: Categorias del fold de test, en 1-hot encoding.
    """
    # entrenar 1 clasificador por categoria usando "one vs. rest", usamos esto para calcular AUC
    classificador_por_clase = OneVsRestClassifier(clasificador)
    # targets_preds_por_clase es una matriz donde cada fila es un vector, y cada columna es el score del clasifcador para cada categoria para la fila correspondiente de test_fold_selected
    targets_preds_por_clase = classificador_por_clase.fit(train_fold_selected, train_targets_binarios_por_clase).predict(test_fold_selected)
    for idx_clase, valor_auc in _calcular_auc_por_clase(test_targets_binarios_por_clase, targets_preds_por_clase).items():
        print("\tAUC para la clase #{} ({}) = {}".format(idx_clase, idx_a_clase[idx_clase], valor_auc))

def pesos_de_features(score_fn, train_fold, train_targets_fold) -> np.ndarray:
    scores =  np.empty((train_fold.shape[1]),dtype=np.float)
    for i in range(0,train_fold.shape[1]):
        scores[i] = score_fn(train_fold[:,i],train_targets_fold)[0]
    return scores

def imprimir_features_con_pesos(score_fn, train_fold, train_targets_fold, nombres_features, top_n=-1):
   """
   Esta funcion evalua que tan bien cada columna de un dataset sirve para clasificar ese dataset.
   :param score_fn: una funcion que pueda tomar una columna de feature y la columna de categoria, y calcular un score que mida que tan bien esa columna predice las categorias. Puede ser cualquier funcion dentro de sklearn.feature_selection como chi2, mutual_info_classif, o relief (si agregan relief con pip install sklearn-relief)
   :train_target_fold: una matriz con columnas a evaluar, excluyend la columna de categoria de cada fila.
   :train_targets_fold: un arreglo con el valor  categoria de  cada fila en :train_target_fold.
   :nombre_features: Los nombres de c/columna en train_target_fold.
   :top_n: cuantos de los mejores scores imprimir. -1 imprime todos.
   """
    pesos_features = pesos_de_features(score_fn, train_fold, train_targets_fold)
    # conseguir los indices que ordenarian a "pesos". Como argsort solo ordena en orden ascendente, damos vuelta el arreglo
    indice_orden_desc_pesos = np.argsort(pesos_features)[::-1]
    if top_n == -1:
        top_n = train_fold.shape[1]
    for i in range(0,top_n):
        print(nombres_features[indice_orden_desc_pesos[i]],'\t',pesos_features[indice_orden_desc_pesos[i]])


def nombres_features_seleccionadas(selector_features, nombres_features):
    """
    Esta funcion retorna los nombres de las columnas seleccionadas como mejores por selector_features.
    :param  selector_features: Una funcion de sklearn que puede evaluar los scores de columna y seleccionar las mejores. Puede ser SelectKBest, GenericUnivariateSelect o SelectPercentile.
    :param  nombres_features: Una lista de nombres de columnas. El orden de las columnas tiene que ser el mismo que el de la matriz con la que se evaluo a selector_features.
    :return new_features: Una lista de nombres de features que se corresponde con las seleccionadas por selector_features.
    """
    cols = selector_features.get_support()
    new_features = []
    for selected, feature in zip(cols, nombres_features):
        if selected:
            new_features.append(feature)
    return new_features


# leer dataset
vectores = joblib.load(VECTORS_FILE)
nombres_targets = joblib.load(TARGETS_FILE)
nombres_features = joblib.load(FEATURE_NAMES_FILE)

# pasar categorias a numeros (1ra categoria = 0, 2da categoria = 1, etc)
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(nombres_targets)

# idx_a_clase es un diccionario indice de categoria -> nombre de categoria
idx_a_clase = label_encoder.classes_

# cantidad de categorias distintas que tenemos en el conj. de entrenamiento
n_categorias = len(idx_a_clase)

# el clasificador que vamos a usar
clasificador = SVC(kernel='linear', probability=True)

# cantidad maxima de features que seleccionara el extractor de features
MAX_FEATURES=150
# cantida de folds a usar en cross-val
CANT_FOLDS_CV=5

# transformar los targets en N columnas, 1 por cada categoria, donde la categoria correcta tiene un 1 y todas las demas columnas en esa fila tienen 0.
# Dado que AUC se calcula sobre 2 categorias, Usamos esto luego para calcular 1 AUC por cada categoria
targets_binarios_por_clase = label_binarize(targets, classes=range(0, n_categorias))

# hacer cross-validation
n_fold = 1
accuracy_promedio = 0


for train_index, test_index in StratifiedKFold(n_splits=CANT_FOLDS_CV, random_state=None, shuffle=True).split(vectores, targets):
        # armar folds de entrenamiento para CV
        train_fold = vectores[train_index]
        train_targets_fold = targets[train_index]
        # armar fold de test para CV
        test_fold = vectores[test_index]
        test_targets_fold = targets[test_index]

        imprimir_features_con_pesos(chi2,train_fold, train_targets_fold, nombres_features, MAX_FEATURES)

        # seleccionar features a partir de los folds de entrenamiento
        selector_features = SelectKBest(score_func=chi2, k=MAX_FEATURES)
        selector_features.fit(train_fold, train_targets_fold)

        # dejar en el fold de entrenamiento solo las features seleccionadas con el fold de entrenamiento
        train_fold_selected = selector_features.transform(train_fold)

        # dejar en el fold de test solo las features seleccionadas con el fold de entrenamiento
        test_fold_selected = selector_features.transform(test_fold)
        selector_features.get_support()
        # clasificar el fold de test
        preds_fold = clasificador.fit(train_fold_selected, train_targets_fold).predict(test_fold_selected)

        print("FOLD #{}, # instancias train = {}, # instancias test = {}".format(n_fold, train_fold.shape[0], test_fold_selected.shape[0]))
        print("FEATURES SELECCIONADAS:")
        print(nombres_features_seleccionadas(selector_features, nombres_features))
        # evaluar accuracy comparando las categorias reales con las predichas
        accuracy_fold = accuracy_score(test_targets_fold, preds_fold)
        accuracy_promedio += accuracy_fold
        print("Accuracy del fold #{} = {}".format(n_fold, accuracy_fold))

        # evaluar AUC, 1 AUC para cada categoria
        calcular_e_imprimir_auc(clasificador, train_fold_selected, targets_binarios_por_clase[train_index], test_fold_selected, targets_binarios_por_clase[test_index])

        print("\tMatriz de confusion (filas=real, columnas=prediccion):")
        mat_conf = confusion_matrix(test_targets_fold, preds_fold)
        print(mat_conf)
        n_fold += 1

print("\nAccuracy promedio = {}".format(accuracy_promedio / n_folds))
