from sklearn.preprocessing import StandardScaler

## Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
import sys,os
from src.utils import save_objects,load_object

from dataclasses import dataclass

@dataclass
## Data Transformation config class 
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')
        except Exception as e:
            raise CustomException(e,sys)