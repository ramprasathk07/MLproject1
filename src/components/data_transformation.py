from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
#==============================================================================
import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#==============================================================================


class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor_obj.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_features =['reading_score', 'writing_score']

            categorical_features = ['gender', 'race_ethnicity', 
                                    'parental_level_of_education', 
                                    'lunch', 'test_preparation_course']
            
            num_pipe=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('std_scaler',StandardScaler())
            ])

            cat_pipe=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('std_scaler',StandardScaler())
            ])

            logging.info("Encoding for features done") 

            preprocessor=ColumnTransformer([
                ('num',num_pipe,numerical_features),
                ('cat',cat_pipe,categorical_features)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException("Error in data transformation",e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Data read successfully")

            preprocessor=self.get_data_transformer_obj()

            logging.info("Preprocessor obtained successfully")

            traget_col=['math_score']

            X_train=train_df.drop(traget_col,axis=1)
            y_train=train_df[traget_col]

            X_test=test_df.drop(traget_col,axis=1)
            y_test=test_df[traget_col]

            logging.info("Data split successfully")

            X_train_arr=preprocessor.fit_transform(X_train)

            X_test_arr=preprocessor.fit_transform(X_test)

            train_arr=np.c_[X_train_arr,np.array(y_train)]
            test_arr=np.c_[X_test_arr,np.array(y_test)] 

            logging.info("Done data transformation") 

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor

            )
            logging.info("Saved preprocessor obj")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )
        
        except Exception as e:
            raise CustomException(e,sys)
