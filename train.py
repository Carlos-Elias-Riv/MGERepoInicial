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
import argparse
import threading

class TimeoutException(Exception):
    pass

def dump_with_timeout(clf, path_for_model, timeout=180):
    """
    Guarda un modelo en un archivo con un tiempo límite.
    """
    def target():
        try:
            joblib.dump(clf, path_for_model)
        except Exception as e:
            global exception
            exception = e

    exception = None
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)  # Wait for 'timeout' seconds

    if thread.is_alive():
        raise TimeoutException(f"joblib.dump took longer than {timeout} seconds and was forcefully stopped.")

    if exception:
        raise exception

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


def train_and_save_model(path_for_model="data/model/model.pkl"):
    """
    Entrena el modelo y lo guarda en un archivo.

    Usa la función `construct_model` para entrenar el modelo y lo guarda 
    en `data/model/model.pkl` utilizando `joblib`.
    """
    clf = construct_model()
    try: 
        dump_with_timeout(clf, path_for_model)
    except TimeoutException as e:
        print("timeout: ", e)
    except Exception as e:
        print("other exception: ", e)
    
def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo')
    parser.add_argument("path_for_model", type = str, help="Ruta para cargar el modelo")

    args = parser.parse_args()
    
    train_and_save_model(args.path_for_model)   
        
if __name__ == "__main__":
    main()
