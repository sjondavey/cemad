import pytest
import os
from cemad_rag.cemad_index import CEMADIndex
from regulations_rag.standard_regulation_index import load_parquet_data, append_parquet_data
from regulations_rag.regulation_reader import load_csv_data, append_csv_data

import pandas as pd

def check_definitions(path_to_definitions_as_parquet_file):
    df_definitions_all = pd.read_parquet(path_to_definitions_as_parquet_file, engine='pyarrow')
    assert len(df_definitions_all.columns) == 3

    expected_columns = ['definition', 'source', 'embedding']
    for column_heading in df_definitions_all.columns:
        assert column_heading in expected_columns

    assert not df_definitions_all.isna().any().any()

def check_index(path_to_index_as_parquet_file):
    df_text_all = pd.read_parquet(path_to_index_as_parquet_file, engine='pyarrow')
    assert len(df_text_all.columns) == 4
    expected_columns = ["section_reference", "text", "source", "embedding"]
    for column_heading in df_text_all.columns:
        assert column_heading in expected_columns


def test_data():
    # Make sure that when you load the manual, there are no NaN values
    manual = load_csv_data("./inputs/ad_manual.csv")
    assert not manual.isna().any().any()

    path_to_definitions_as_parquet_file = "./inputs/ad_definitions.parquet"
    check_definitions(path_to_definitions_as_parquet_file)
    path_to_definitions_as_parquet_file = "./inputs/ad_definitions_plus.parquet"
    check_definitions(path_to_definitions_as_parquet_file)

    path_to_index_as_parquet_file = "./inputs/ad_index.parquet"
    check_index(path_to_index_as_parquet_file)
    path_to_index_as_parquet_file = "./inputs/ad_index_plus.parquet"
    check_index(path_to_index_as_parquet_file)


def test_load_csv_data():
    df_document = load_csv_data("./inputs/ad_manual.csv")
    assert not df_document.isna().any().any()

def test_append_csv_data():
    df_document = load_csv_data("./inputs/ad_manual.csv")
    l = len(df_document)
    df_document = append_csv_data("", df_document)
    assert len(df_document) == l
    assert not df_document.isna().any().any()
    df_document = append_csv_data("./inputs/ad_manual_plus.csv", df_document)
    assert len(df_document) > l
    assert not df_document.isna().any().any()

def test_load_parquet_data():
    df_text_all = load_parquet_data("./inputs/ad_definitions.parquet")
    assert not df_text_all.isna().any().any()

def test_append_parquet_data():
    df_text_all = load_parquet_data("./inputs/ad_definitions.parquet")
    l = len(df_text_all)
    df_text_all = append_parquet_data("", df_text_all)
    assert len(df_text_all) == l
    df_text_all = append_parquet_data("./inputs/ad_definitions_plus.parquet", df_text_all)
    assert len(df_text_all) > l
    assert not df_text_all.isna().any().any()

    key = key = os.getenv('excon_encryption_key')
    df_index = load_parquet_data("./inputs/ad_index.parquet", key)
    assert df_index.iloc[0]['text'] == 'Legal context' # check it is decrypted
    df_index_plus = append_parquet_data("./inputs/ad_index_plus.parquet", df_index, key)
    assert df_index_plus.iloc[-1]['text'] == 'What is the current version date of the manual?' # check it is decrypted



class TestCEMADIndex:
    key = key = os.getenv('excon_encryption_key')
    data = CEMADIndex(key)


    def test_construction(self):
        assert True




