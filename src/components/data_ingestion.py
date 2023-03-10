
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  
#=====================================================
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig,DataTransformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
@dataclass
class dataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=dataIngestionConfig()
        
    def init_data_ingestion(self):
        # Read the data from the source like DB

        logging.info("Reading the data from the source")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Data read successfully")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Data save successful")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data train test successful")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                

            )
        except Exception as e:
            raise CustomException("Error in data ingestion",e)
        

if __name__=="__main__":
    di=DataIngestion()
    train,test=di.init_data_ingestion()

    dt=DataTransformation()
    train_arr,test_arr,_=dt.initiate_data_transformation('D:\DOCS\MLPgit/artifacts/train.csv','D:\DOCS\MLPgit/artifacts/test.csv')

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


    