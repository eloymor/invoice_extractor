import json
import os
import re
import pandas as pd
import numpy as np
from facturas.config import (key_words, items_to_find, items_to_find_path, taxes_values, tax_check_words,
                             mapping_items_holded)


def convert_to_json(text: str):

    try:
        return json.loads(text)
    except:
        return None


def context_words(keywords: dict[str, str]) -> str:

    return ", ".join([f"{k} is equal to {v}" for k, v in keywords.items()])


def create_main_items_df(items: dict[str, str]) -> pd.DataFrame:
    # Wrap the items dictionary in a list to create a DataFrame with a single row
    # This avoids the "If using all scalar values, you must pass an index" error
    return pd.DataFrame([items])


def load_key_words(file_path: str) -> dict[str, str]:
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(key_words, file)

    with open(file_path, 'r') as file:
        return json.load(file)


def save_key_words(file_path: str, keywords: dict[str, str]):
    with open(file_path, 'w') as file:
        json.dump(keywords, file)


def load_items_to_find(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(items_to_find, file)

    with open(file_path, 'r') as file:
        return json.load(file)

def save_items_to_find(file_path: str, items: list[str]):
    with open(file_path, 'w') as file:
        json.dump(items, file)


def convert_to_float(number_text: str | int | float) -> float:
    """
    Converts a possible number string to a float.
    :param number_text: string to convert
    :return: float if conversion is successful, otherwise 0.0
    """
    if isinstance(number_text, float) or isinstance(number_text, int):
        return float(number_text)
    try:
        number_text = re.sub(r"[^\d.,]", "", number_text)
        number_text = number_text.replace(",", ".")
        return float(number_text)
    except ValueError:
        return 0.0


def check_iva(df: pd.DataFrame, iva: int):

    df = df.copy()
    col_name: str = f'invoice_iva_{iva}'
    if col_name in df.columns:
        if df[col_name].iloc[0] is not None:
            if isinstance(df[col_name].iloc[0], np.float64):
                df.loc[0, col_name] = convert_to_float(df.loc[0, col_name].item())
                if df[col_name].iloc[0].item() <= 0 or df[col_name].iloc[0].item() == iva:
                    df.loc[0, col_name] = 0.0
            else:
                df.loc[0, col_name] = convert_to_float(df.loc[0, col_name])
                if df[col_name].iloc[0] <= 0 or df[col_name].iloc[0] == iva:
                    df.loc[0, col_name] = 0.0

        else:
            df.loc[0, col_name] = 0.0
    else:
        df.loc[0, col_name] = 0.0

    return df


def update_main_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updates and formats the main DataFrame.
    :param df: pd.DataFrame
    :return: updated pd.DataFrame
    """
    df = df.copy()

    new_cols: list[str] = load_items_to_find(items_to_find_path)

    if len(df.columns) == 12:
        df.rename(dict(zip(list(df.columns), new_cols)), axis=1, inplace=True)
    elif len(df.columns) > 12:
        df.drop(df.columns[12:], axis=1, inplace=True)
        df.rename(dict(zip(list(df.columns), new_cols)), axis=1, inplace=True)

    for tax in taxes_values:
        df = check_iva(df, tax)

    return df


# TODO
def update_items_df(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df_items_cols: list[str] = [col.lower() for col in df.columns]
    if any(tax_word.lower() in tax_check_words for tax_word in df_items_cols):
        for tax in tax_check_words:
            tax = tax.lower()
            if tax in df_items_cols:
                df.groupby(df_items_cols).sum()


    pass


def retrieve_items(df: pd.DataFrame) -> dict[str, str | float]:
    df = df.copy()
    items: dict[str, str | float] = {}
    for col_name in df.columns:
        value = df[col_name].iloc[0]
        if isinstance(value, str) and value.strip() != "":
            items[col_name] = value
        elif isinstance(value, float):
            try:
                items[col_name] = value.item()
            except AttributeError:
                items[col_name] = value
        else:
            items[col_name] = ""
    return items


def format_items_holded(items: dict, mapping: dict) -> pd.DataFrame:

    dataframes: list[pd.DataFrame] = []
    iva_types: list[str] = []
    for key, value in items.items():
        if "iva" in key.lower() and value > 0:
            iva_types.append(key)

    for iva_type in iva_types:
        formatted_items: dict[str, str | float] = {}
        for key, value in mapping.items():
            if key == "IVA":
                iva_pct = int(iva_type.split("_")[-1])
                formatted_items[key] = iva_pct
            elif key == "unit_price":
                formatted_items[key] = items[iva_type]
            elif key == "units":
                formatted_items[key] = 1
            elif key == "invoice_number":
                formatted_items[key] = items["invoice_number"]
            elif key == "invoice_date":
                formatted_items[key] = items["invoice_date"]
            elif key == "issuer_name":
                formatted_items[key] = items["issuer_name"]
            elif key == "issuer_NIF":
                formatted_items[key] = items["issuer_NIF"]
            elif key == "issuer_address":
                formatted_items[key] = items["issuer_address"]
            elif key == "issuer_city":
                formatted_items[key] = items["issuer_city"]
            elif key == "issuer_postal_code":
                formatted_items[key] = items["issuer_postal_code"]
            elif key == "country":
                formatted_items[key] = "España"
            elif key == "discount":
                formatted_items[key] = 0
            elif key == "retention":
                formatted_items[key] = 0
            elif key == "currency":
                formatted_items[key] = "EUR"
            elif key == "description":
                formatted_items[key] = f"factura {items['invoice_number']} de {items['issuer_name']}"
            elif key == "item":
                formatted_items[key] = f"parte iva {iva_type.split("_")[-1]} %"
            elif key == "item_description":
                formatted_items[key] = f"parte iva {iva_type.split("_")[-1]} %"
            else:
                formatted_items[key] = ""

        new_dict = {}
        for old_key, value in formatted_items.items():
            new_key = mapping.get(old_key, old_key)  # Use new key if exists, else keep old key
            new_dict[new_key] = value

        df = pd.DataFrame([new_dict])
        dataframes.append(df)

    if len(dataframes) > 1:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return dataframes[0]


def get_holded_format(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    updated_df: pd.DataFrame = update_main_df(df)
    items: dict[str, str | float] = retrieve_items(updated_df)
    formatted_df: pd.DataFrame = format_items_holded(items, mapping_items_holded)

    return formatted_df