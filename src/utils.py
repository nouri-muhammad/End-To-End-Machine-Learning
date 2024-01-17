import dill 
import numpy as np 
import pandas as pd 
import os 
import sys
from src.exception import CustomException


def null_row_dropper(file_obj, columns):
    file_obj = file_obj.dropna(subset=columns).reset_index(drop=True)
    return file_obj


def duplicate_dropper(file_obj):
    duplicates = file_obj.duplicated()
    if duplicates.sum() > 0:
        file_obj = file_obj.drop_duplicates(keep='first').reset_index(drop=True)
    return file_obj


def invalid_data_dropper(file_obj):
    file_obj = file_obj[file_obj['EngineV'] < 6.5].reset_index(drop=True)
    return file_obj


def outlier_detection(df: pd.DataFrame, columns: list, threshold: float):
    """
    This function takes a dataframe, columns we need delete outliers from and a threshold for z-score then
    drops any rows from data that its value is above the threshold
    """
    upper_limit = threshold
    lower_limit = (-1) * threshold
    for column in columns:
        df['z_score'] = (df[column] - df[column].mean()) / df[column].std()
        df = df[(df['z_score'] < upper_limit) & (df['z_score'] > lower_limit)]
        df = df.drop(columns='z_score', axis=1)
    return df
