## Importing library
import numpy as numpy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from src.logger import logging
from src.exception import CustomException
from src.utils import save_objects,load_object,evalute_model

from dataclasses import dataclass
import sys,os

## 
@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_confif = ModelTrainerConfig()
