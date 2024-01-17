import numpy as np 
import pandas as pd 
import os 
import sys
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('processed_data', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
        This function is responsible for data transformation based on the data
        """
        try:
            numerical_columns = ['Mileage', 'EngineV', 'Year']
            categorical_columns = ['Brand', 'Body', 'Engine_Type']

            num_pipeline = Pipeline(
                steps=[
                    # ("Imputer", SimpleImputer(strategy="median")), 
                    # replace all null values in numeric columns with median of the respective column if we decide not to remove the row
                    ("Scaler", StandardScaler())
                    # scaling the numerical values
                ]
            )
            logging.info("Numerical Columns' Scaling Completed")
            cat_pipeline = Pipeline(
                steps=[
                    # ("Imputer", SimpleImputer(strategy="most_frequent")),
                    # replace all null values in categorical columns with most frequent value in the respective column if we decide not to remove the null values
                    ("OneHotEncoder", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical Columns' Encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read The Train and Test Data")

            logging.info("Getting Preprocessor Object")
            preprocessor_obj = self.get_data_transformer_obj()
            target_column_name = 'Price'
            numerical_columns = ['Mileage', 'EngineV', 'Year']
            categorical_columns = ['Brand', 'Body', 'Engine_Type']
            
            logging.info("Splitting Train and Test Datasets into Feature and Target")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Apply Preprocessing Object on Train and Test Data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Save Preprocessing Object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
