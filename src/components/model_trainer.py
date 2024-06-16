from cmath import inf
import imp
import os, sys

from dataclasses import dataclass
from tabnanny import verbose
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join('artifacts', "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "random forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "catBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            
            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models = models)
            
            best_model_score = float('-inf')
            best_model_name = ''
            
            for keys in model_report:
                if model_report[keys] > best_model_score:
                    best_model_score = model_report[keys]
                    best_model_name = keys
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise customException("no best model found")
            
            logging.info("best model found on both training and testing")
            
            save_object(self.model_trainer_config.trained_model_file, best_model)
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
            
        except Exception as e:
            raise customException(e,sys)
        
