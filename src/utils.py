import dill 
import numpy as np 
import pandas as pd 
import os 
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file=file_path, mode="wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


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
        df = df[(df['z_score'] < upper_limit) & (df['z_score'] > lower_limit)].reset_index(drop=True)
        df = df.drop(columns='z_score', axis=1)
    return df


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        fitters = {}
        for name, model in models.items():
            regr = model
            gs = GridSearchCV(regr, params[name], cv=3)
            gs.fit(x_train ,y_train)

            regr.set_params(**gs.best_params_)
            regr.fit(x_train ,y_train)
            
            y_test_pred = regr.predict(x_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
            fitters[name] = regr
            logging.info(f"Model: {name} Uptimization is Done!")

        return report, fitters
    except Exception as e:
        raise CustomException(e, sys)
