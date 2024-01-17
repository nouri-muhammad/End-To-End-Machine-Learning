import pandas as pd 
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import null_row_dropper, duplicate_dropper, invalid_data_dropper, outlier_detection


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('processed_data', 'train.csv')
    test_data_path :str=os.path.join('processed_data', 'test.csv')
    raw_data_path :str=os.path.join('processed_data', 'raw_data.csv')


class DataIngestion:

    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function reads data from database or from a file
        """
        logging.info("Initiated Data Ingestion")
        try:
            data=pd.read_csv("data/UsedCars.csv")
            df=data.copy()
            logging.info("Read the dataset")
            # here we read the data from a csv file
            # by chaanging this part we can read the data from other sources

            logging.info("Dropping rows with null values")
            null_drop_columns = ['EngineV', 'Price']
            df = null_row_dropper(df, null_drop_columns)

            logging.info("Dropping Duplicate Data")
            df = duplicate_dropper(df)

            logging.info("Dropping Invalid Data From EngineV Column")
            df = invalid_data_dropper(df)

            logging.info("Dropping Outliers")
            columns = ['Price', 'Year', 'Mileage']
            df = outlier_detection(df=df, columns=columns, threshold=3)

            # creating the folder for saving train and test datasets
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # exists_ok=True means if the folder exists do noting, otherwise create it
            
            # saving the raw, train and test datasets in the created folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train Test Split initiated")
            train_set, test_set =  train_test_split(df, test_size=0.2, random_state=91)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion Process is Completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
