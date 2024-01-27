import os,sys,pickle
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def save_objects(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info('Error occured while saving model')
        raise CustomException(e,sys)


def evalute_model(X_train_scaled,y_train,X_test_scaled,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model= list(models.values())[i]
            #Train Model
            model.fit(X_train_scaled,y_train)

            #predicting Testing Data
            y_pred=model.predict(X_test_scaled)

            #Getting Accuracy Score
            test_model_score = accuracy_score(y_test,y_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        logging.info('Error occured during model evaluation')
        raise CustomException(e,sys)



def load_object(file_path):
    try:
        with open(file_name,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)