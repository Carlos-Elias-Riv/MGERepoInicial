"""
Módulo de Preprocesamiento de Datos para Predicción de Ventas

Este módulo contiene funciones para la carga, limpieza y preprocesamiento de datos 
de ventas. Además, permite la agregación y fusión de datos para su uso en modelos 
de predicción.

Dependencias:
    - pandas
    - os
    - tqdm

Funciones:
    - load_data(path): Carga archivos CSV en dataframes de pandas.
    - preprocess_sales_data(sales_train): Convierte las fechas al formato de pandas y extrae el mes.
    - remove_outliers_iqr(data, col): Filtra valores atípicos usando el rango intercuartílico (IQR).
    - clean_sales_data(sales_train): Elimina valores negativos y aplica la detección de outliers.
    - check_price_variation(clean_data): Revisa la variación de precios por artículo.
    - prepare_aggregated_data(clean_data): Agrega los datos por mes, tienda e ítem.
    - save_data(items_info, grouped_data): Guarda los datos procesados en archivos CSV.
    - merge_data(): Une los datos procesados en un solo CSV para entrenamiento del modelo.

Uso:
    Importa el módulo y llama a las funciones en el orden adecuado para cargar, limpiar, 
    procesar y guardar los datos.

    Ejemplo:
        data = load_data("../data")
        sales_data = preprocess_sales_data(data["sales_train"])
        cleaned_data = clean_sales_data(sales_data)
        check_price_variation(cleaned_data)
        items_info, grouped_data = prepare_aggregated_data(cleaned_data)
        save_data(items_info, grouped_data)
        merge_data()
"""

import os
from tqdm import tqdm
import pandas as pd

# Define the data path
PATHTAREA = "data"


def load_data(path):
    """Carga los CSVS en pandas dataframes."""
    files = ["item_categories.csv", "items.csv", "sales_train.csv",
             "sample_submission.csv", "shops.csv", "test.csv"]

    data = {file.split('.', maxsplit=1)[0]: pd.read_csv(
        os.path.join(path, file)) for file in files}

    return data


def preprocess_sales_data(sales_train):
    """Preprocesamiento para fechas de las ventas."""
    sales_train['date'] = pd.to_datetime(
        sales_train['date'], format='%d.%m.%Y')
    sales_train['month'] = sales_train['date'].dt.month
    return sales_train


def remove_outliers_iqr(data, col):
    """Quitando outliers usando el rango intercuartilico."""
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data[col] >= lower) & (data[col] <= upper)]


def clean_sales_data(sales_train):
    """Filtrar valores negativos y outliers."""
    non_negative_data = sales_train[(sales_train['item_cnt_day'] > 0) &
                                    (sales_train['item_price'] > 0)]
    filtered_item_cnt = remove_outliers_iqr(non_negative_data, "item_cnt_day")
    cleaned_data = remove_outliers_iqr(filtered_item_cnt, "item_price")
    return cleaned_data


def check_price_variation(clean_data):
    """Revision de variacion de precios."""
    price_variation = clean_data.groupby("item_id")["item_price"].nunique()
    items_with_price_changes = price_variation[price_variation > 1]

    if not items_with_price_changes.empty:
        print("Items with price changes detected:")
        print(items_with_price_changes)
    else:
        print("All items have consistent prices.")


def prepare_aggregated_data(clean_data):
    """Preparacion de datos agregados. Es decir agrupar por mes, tienda e item."""
    items_info = clean_data[['item_id', 'item_price']]

    interested_data = clean_data[['date_block_num',
                                  'month', 'shop_id', 'item_id', 'item_cnt_day']]
    grouped_data = interested_data.groupby(
        ['date_block_num', 'shop_id', 'item_id', 'month']).sum().reset_index()

    return items_info, grouped_data


def save_data(items_info, grouped_data):
    """Guarda los datos en CSVs."""
    grouped_data.to_csv("grouped_data.csv", index=False)
    items_info.to_csv("items_info.csv", index=False)


def merge_data():
    """Une los datos en un solo CSV. Este CSV se usara para entrenar el modelo."""
    grouped_data_parts = pd.read_csv("grouped_data.csv", chunksize=10000)
    items_info_parts = pd.read_csv("items_info.csv", chunksize=10000)

    df_list = []
    for chunk_item in tqdm(items_info_parts, desc="Processing item info chunks"):
        for chunk_grouped in grouped_data_parts:
            df_list.append(pd.merge(chunk_item, chunk_grouped,
                           how="inner", on="item_id"))

    joined_data = pd.concat(df_list)
    joined_data.to_csv("joined_data.csv", index=False)
