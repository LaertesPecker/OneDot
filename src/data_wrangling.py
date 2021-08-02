import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Any
from geopy.geocoders import Nominatim


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def pre_processing(
    df: pd.DataFrame,
    key: str = "Attribute Names",
    value: str = "Attribute Values",
    id: str = "ID",
) -> pd.DataFrame:
    """Transform the supplier data to achieve the same granularity as the target data, returns a DataFrame

    Args:
        df (pd.DataFrame): DataFrame with the supplier data
        key (str, optional): Name of the column with the columns names to unstack. Defaults to "Attribute Names"
        value (str, optional): Name of the column with the values to unstack. Defaults to "Attribute Values"
        id (str, optional): Name of the column that identifies each product. Defaults to "ID".

    Returns:
        pd.DataFrame: DataFrame with the columns and values from key and value unstacked and merged with original dataframe
    """
    try:
        df1 = df.set_index([id, key])[value].unstack()
        df2 = df.loc[:, ~df.columns.isin([key, value])].groupby(id).first()
    except Exception:
        logging.exception("An unexpected error occured while pre-processing")
    else:
        logging.info("Pre-processing done correctly")
        return df1.merge(df2, on=id)


def translate(
    df: pd.DataFrame,
    col: str,
    trans_filepath: str = "./data/translations.json",
    fill: str = "Other",
) -> pd.DataFrame:
    """Translate colors from dutch to english using a json dict

    Args:
        df (pd.DataFrame): Original DataFrame.
        col (Union[str,List(str)]): Columns with the values to translate.
        trans_filepath (str, optional): Filepath to the json containing the translations. Defaults to 'colors_translation.json'.
        fill (str, optional): Fill value when translation not in trans_filepath file. Defaults to "Other".

    Returns:
        pd.DataFrame: DataFrame with the column "col" translated
    """
    try:
        with open(trans_filepath) as json_file:
            d = json.load(json_file)
        df[col] = df[col].map(d).fillna(fill)
    except Exception:
        logging.exception("An unexpected error occured while normalizing colors")
    else:
        logging.info(f"{col.title()} Translated correctly")
        return df


def normalize(
    df: pd.DataFrame, col: Dict[str, Any], fill: str = "null"
) -> pd.DataFrame:
    """Changes types, renames data and fills missing values

    Args:
        df (pd.DataFrame): Original DataFrame
        col (Dict[str, List[Any]]): Column with the values to change type
        fill (str): values to fill na's
    
    Returns:
        pd.DataFrame: DataFrame with the new column and new types
    """
    try:
        for new_column_name, values in col.items():
            new_type = values[1]
            old_column_name = values[0]
            if old_column_name is not None:
                if len(values) == 3:
                    df[old_column_name] = df[old_column_name].apply(values[2])
                    df[new_column_name] = df[old_column_name].astype(new_type)
                else:
                    df[new_column_name] = df[old_column_name].astype(new_type)
                if new_column_name != old_column_name:
                    df.drop(columns=old_column_name, inplace=True)
            else:
                if len(values) == 3:
                    df[new_column_name] = values[2]
                    df[new_column_name] = df[new_column_name].astype(new_type)
            df[new_column_name].fillna({new_column_name: fill}, inplace=True)
    except Exception:
        logging.exception("An unexpected error occured while normalizing")
    else:
        logging.info("Normalization done correctly")
        return df


def get_country_codes(
    df: pd.DataFrame, col: str, col_name: str = "country"
) -> pd.DataFrame:
    """Get country codes from country names

    Args:
        df (pd.DataFrame): Original DataFrame
        col (str): Column with the country names
        col_name (str, optional): Name of the new column with the country codes. Defaults to "country".

    Returns:
        pd.DataFrame: DataFrame with country codes
    """
    try:
        geolocator = Nominatim(user_agent="get_city")
        city_dict = {}
        for city in df[col].unique():
            city_dict[city] = (
                geolocator.geocode(city, addressdetails=True)
                .raw["address"]["country_code"]
                .upper()
            )
        df[col_name] = df[col].map(city_dict)
    except Exception:
        logging.exception(
            "An unexpected error occured while retrieving country codes form cities"
        )
    else:
        logging.info("Country code extraction correctly")
        return df


if __name__ == "__main__":
    logging.info("Started data wrangling")
    # Pre-processing
    try:
        data = pd.read_json("./data/supplier_car.json", lines=True)
    except Exception:
        logging.exception("An unexpected error occured while reading file")
    pre = pre_processing(data)
    # Normalization
    norm = pre.copy(deep=True)
    norm = translate(norm, "BodyColorText")
    norm = translate(norm, "BodyTypeText")
    norm = translate(norm, "ConditionTypeText")
    norm = get_country_codes(norm, "City")
    norm_dict = {
        "carType": ["BodyTypeText", "category"],
        "color": ["BodyColorText", "category"],
        "condition": ["ConditionTypeText", "category"],
        "currency": [None, "category", "null"],
        "drive": [None, "category", "null"],
        "city": ["City", "category"],
        "country": ["country", "category"],
        "make": [
            "MakeText",
            "category",
            lambda x: x.title()
            if x not in ["RUF", "NSU", "BMW", "VW", "MG", "PGO", "MINI", "AGM"]
            else x,
        ],
        "manufacture_year": ["FirstRegYear", "h"],
        "mileage": ["Km", np.float32],
        "mileage_unit": [None, "category", "kilometer"],
        "model": ["ModelText", "O"],
        "model_variant": ["TypeName", "O"],
        "price_on_request": [None, "category", "false"],
        "type": [None, "category", "car"],
        "zip": [None, "category", "null"],
        "manufacture_month": ["FirstRegMonth", "h"],
        "fuel_consumption_unit": [
            "ConsumptionTotalText",
            "category",
            lambda x: "l_km_consumption" if len(x.split(" ")) > 1 else "null",
        ],
    }
    norm = normalize(norm, norm_dict)

    # Integration
    try:
        integration = norm.loc[:, norm_dict.keys()]
        with pd.ExcelWriter("output.xlsx") as writer:
            pre.to_excel(writer, sheet_name="Pre-Processing", index=False)
            norm.to_excel(writer, sheet_name="Normalization", index=False)
            integration.to_excel(writer, sheet_name="Integration", index=False)
    except Exception:
        logging.exception("An unexpected error occured while exporting files to excel")
    logging.info("Finished data wrangling")

