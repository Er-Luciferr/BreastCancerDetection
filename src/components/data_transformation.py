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

            cols = ['mean radius', 'mean texture', 'mean smoothness', 'mean compactness',
       'mean concavity', 'mean concave points', 'mean symmetry',
       'mean fractal dimension', 'radius error', 'texture error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst texture', 'worst smoothness', 'worst compactness',
       'worst concavity', 'worst concave points', 'worst symmetry',
       'worst fractal dimension', 'pc1', 'pc2','target']

            ## Creating Pipeline
            pipeline= Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('pipeline',pipeline,cols)
            ])

            return preprocessor
            logging.info('Data Transformation Done')
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path , test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Reading of train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj=self.get_data_transformation_object()


            ## features into independent and dependent features

            input_train_df = train_df
            output_train_df = train_df['target']

            input_test_df = test_df
            output_test_df = test_df['target']

            ## Data Transformation

            input_train_df_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_df_arr= preprocessor_obj.transform(input_test_df)
            logging.info('Applying preprocessor object on train and test set has been done')

            train_arr = np.c_[input_train_df_arr,np.array(output_train_df)]
            test_arr=np.c_[input_test_df_arr,np.array(output_test_df)]

            save_objects(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
