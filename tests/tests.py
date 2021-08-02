import pytest
import pandas as pd
import numpy as np

from src import data_wrangling


@pytest.fixture(scope="function")
def read_json(request):
    df = pd.read_json("./tests/supplier_car_test.json", lines=True)
    return df


@pytest.fixture(scope="function")
def preprocess(request):
    df = pd.read_json("./tests/supplier_car_test.json", lines=True)
    pre = data_wrangling.pre_processing(df)
    return pre


def test_pre_check_total_colums(read_json):
    df = data_wrangling.pre_processing(read_json)
    print(read_json["Attribute Names"])
    assert len(df.columns) == 25


def test_preprocessing_check_new_colums(read_json):
    df = data_wrangling.pre_processing(read_json)
    processed_properties = read_json["Attribute Names"].unique()
    assert all(prop in df.columns.tolist() for prop in processed_properties)


def test_preprocessing_check_old_colums(read_json):
    df = data_wrangling.pre_processing(read_json)
    subset = read_json[
        read_json.columns.difference(["Attribute Names", "Attribute Values", "ID"])
    ]
    assert all(oldcols in df.columns.tolist() for oldcols in subset.columns.tolist())


def test_normalization_translation(preprocess):
    col_to_translate = "BodyColorText"
    translations = pd.read_json("./tests/translations_test.json", orient="index")
    trans_df = data_wrangling.translate(
        preprocess, col_to_translate, "./tests/translations_test.json"
    )
    assert all(
        trans in np.append(translations[0].unique(), ("Other"))
        for trans in trans_df[col_to_translate].unique()
    )


def test_normalization_normalization_new_col(preprocess):
    norm_dict = {"color": ["BodyColorText", "category"]}
    norm = data_wrangling.normalize(preprocess, norm_dict)
    assert "color" in norm.columns


def test_normalization_normalization_new_type(preprocess):
    norm_dict = {"color": ["BodyColorText", "category"]}
    norm = data_wrangling.normalize(preprocess, norm_dict)
    assert norm["color"].dtypes == "category"


def test_normalization_normalization_applied_func(preprocess):
    trans_df = data_wrangling.translate(
        preprocess, "BodyColorText", "./tests/translations_test.json"
    )
    norm_dict = {"color": ["BodyColorText", "category", lambda x: x.upper()]}
    norm = data_wrangling.normalize(trans_df, norm_dict)
    assert all(color.isupper() for color in norm["color"])
