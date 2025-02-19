"""
Módulo de Predicción de Ventas

Este módulo carga un modelo de aprendizaje automático previamente entrenado 
y proporciona una función para realizar predicciones de ventas para una 
combinación específica de tienda y artículo.

Dependencias:
    - joblib
    - pandas

Funciones:
    - run_predict(shop_id, item_id): Realiza una predicción de ventas para 
      un artículo específico en una tienda determinada utilizando el modelo cargado.

Uso:
    Importa el módulo y llama a la función `run_predict` con los identificadores 
    de tienda y artículo como argumentos.

    Ejemplo:
        prediction = run_predict(25, 1234)
        print(f"Predicción de ventas: {prediction}")
"""

import joblib

import pandas as pd


clf = joblib.load("data/model/model.pkl")


def run_predict(shop_id, item_id):
    """
    Correr la predicción del modelo dado un item y una tienda.
    Args:
        shop_id (int): Id de la tienda.
        item_id (int): Id del item.
    Returns:
        float: Predicción del modelo
    """
    result = pd.read_csv("data/prep/joined_data.csv")
    result = result[(result['item_id'] == item_id) &
                    (result['shop_id'] == shop_id)]
    mean_price = result['item_price'].mean()
    columns = ['item_id', 'item_price', 'date_block_num', 'shop_id', 'month']
    # 34 y 11 hardcodeados porque son los correspondientes a predecir el siguiente mes
    new_observation = pd.DataFrame([[item_id, mean_price, 34, shop_id, 11]],
                                   columns=columns)

    prediction = clf.predict(new_observation)

    return prediction[0]
