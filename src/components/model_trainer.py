import os,sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

#============================================================================

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    )
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

#============================================================================

@dataclass
class ModelTrainerConfig:
    trained_model_filepath=os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config=config
    def initiate_model_trainer(self,train_arr,test_arr):
        
        try:
            logging.info("Initiating model training")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )
            logging.info("Splitting data into train and test done")
            model={
                'LinearRegression':LinearRegression(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor()

            }
            logging.info("Model training started")
            model_report:dict=evaluate_model(model,X_train,y_train,X_test,y_test)
            best_model_scores=max(sorted(model_report.values()))
            best_model_name=list(model.report.keys())[
                list(model.report.keys()).index(best_model_scores)
            ]
            best_model=model[best_model_name]
            if best_model_scores<0.6:
                raise CustomException("no best model found")
            
            logging.info("Model training completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

                )
            predicted=best_model.predict(X_test)
            return r2_score(predicted,y_test)
        

        except Exception as e:
            raise CustomException("Error in model training",e,sys)
