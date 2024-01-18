import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.components.parameters import params 
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('processed_data', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split Train and Test Data to Feature and Target")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("training and test input data splitted")
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            logging.info("Get Models' Report")
            model_report, fit_model = evaluate_models(
                x_train=x_train, y_train=y_train, 
                x_test=x_test, y_test=y_test,
                models=models, params=params
            )

            logging.info("Best Model found!")
            best_model = max(model_report, key=model_report.get)

            if model_report[best_model] < 0.6:
                raise CustomException("None of the models' performance is satisfactory")
            logging.info(f"Best model was {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = fit_model[best_model]
            )

            logging.info("Initiating Prediction")
            prediction = fit_model[best_model].predict(x_test)
            r2_squared = r2_score(y_test, prediction)

            logging.info("Returning The Result")
            return r2_squared, fit_model[best_model]

        except Exception as e:
            raise CustomException(e, sys)
