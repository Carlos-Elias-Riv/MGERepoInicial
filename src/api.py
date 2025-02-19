"""
API para Predicción de Ventas

Este módulo implementa un servidor FastAPI para realizar predicciones 
utilizando un Random Forest entrenado para predecir la cantidad de artículos
vendidos en un mes para un artículo y una tienda dadas.

Dependencias:
    - fastapi
    - pandas
    - pydantic
    - uvicorn
    - model (módulo propio para cargar el modelo)

Endpoints:
    - GET /predict?shop_id=<int>&item_id=<int> : Retorna la predicción para un item y una tienda.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model import construct_model


def run_predict(shop_id: int, item_id: int) -> float:
    """
    Ejecuta la predicción del modelo dado un identificador de tienda y producto.

    Args:
        shop_id (int): Identificador de la tienda.
        item_id (int): Identificador del producto.

    Returns:
        float: Predicción del modelo.
    """
    result = pd.read_csv("joined_data.csv")
    result = result[(result['item_id'] == item_id) &
                    (result['shop_id'] == shop_id)]
    mean_price = result['item_price'].mean()

    new_observation = pd.DataFrame(
        [[item_id, mean_price, 34, shop_id, 11]],
        columns=['item_id', 'item_price', 'date_block_num', 'shop_id', 'month']
    )

    clf = construct_model()
    prediction = clf.predict(new_observation)

    return prediction[0]


class ModelRequest(BaseModel):
    """
    Modelo de solicitud para la API de predicción.

    Atributos:
        shop_id (int): Identificador de la tienda.
        item_id (int): Identificador del producto.
    """
    shop_id: int
    item_id: int


app = FastAPI()


@app.get("/predict")
def get_model_predict(shop_id: int, item_id: int) -> dict:
    """
    Endpoint para obtener una predicción del modelo.

    Args:
        shop_id (int): Identificador de la tienda.
        item_id (int): Identificador del producto.

    Returns:
        dict: Respuesta con la predicción.
    """
    prediction = run_predict(shop_id, item_id)
    return {'response': prediction}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
