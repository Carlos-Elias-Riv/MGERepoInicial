"""
Módulo de Entrenamiento de Modelo de Predicción de Ventas

Este módulo entrena un modelo de clasificación basado en un Random Forest 
para predecir la cantidad de artículos vendidos en función de diversas características.

Dependencias:
    - pandas
    - joblib
    - sklearn

Funciones:
    - construct_model(): Construye y entrena el modelo de Random Forest.
    - train_and_save_model(): Entrena el modelo y lo guarda en un archivo.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def construct_model():
    """
    Construye y entrena un modelo de clasificación Random Forest.

    Carga los datos preprocesados, selecciona las características y el objetivo,
    divide los datos en conjunto de entrenamiento y prueba, y ajusta el modelo.

    Returns:
        RandomForestClassifier: Modelo entrenado.
    """
    data = pd.read_csv("data/prep/joined_data.csv")
    data['item_count'] = data['item_cnt_day']
    data = data[['item_id', 'item_price', 'date_block_num',
                 'shop_id', "item_count", 'month']]

    target = data['item_count']
    explanatory = data[['item_id', 'item_price',
                        'date_block_num', 'shop_id', 'month']]

    x_train, _, y_train, _ = train_test_split(
        explanatory, target, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)

    return clf


def train_and_save_model():
    """
    Entrena el modelo y lo guarda en un archivo.

    Usa la función `construct_model` para entrenar el modelo y lo guarda 
    en `data/model/model.pkl` utilizando `joblib`.
    """
    clf = construct_model()
    joblib.dump(clf, "data/model/model.pkl")
