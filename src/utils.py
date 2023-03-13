import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
#======================================================================
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(model,X_train,y_train,X_test,y_test):
    try:   
        model_report:dict={}
        for name,regressor in model.items():
            regressor.fit(X_train,y_train)
            y_pred=regressor.predict(X_test)
            r2_scores=r2_score(y_test,y_pred)
            model_report[name]=r2_scores
        return model_report
    except Exception as e:
        raise CustomException(e,sys)