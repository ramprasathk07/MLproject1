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
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
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
                
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor()

            }
            params={
                "LinearRegression": {'fit_intercept': [True, False],'normalize': [True, False]},
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info("Model training started")
            model_report=evaluate_model(model=model,X_train=X_train,
                                             y_train=y_train,X_test=X_test,y_test=y_test,params=params)
            print("\n",model_report,"\n")
            best_model_scores=max(model_report.values())
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_scores)
            ]
            best_model=model[best_model_name]
            logging.info("Model training completed")
            if best_model_scores<0.6:
                raise CustomException("no best model found")
            
            

            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model

                )
            predicted=best_model.predict(X_test)
            
            return r2_score(predicted,y_test)
        

        except Exception as e:
            raise CustomException(e,sys)
