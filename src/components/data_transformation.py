from cmath import e
import os, sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import customException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig :
    preprocessor_path = os.path.join('artifacts' , "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["reading_score","writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            #pipelines
            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("column scaling and encoding done")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise customException(e, sys)
    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("reading of test and tran data done")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column = "math_score"
            numerical_columns = ["reading_score","writing_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info(f"applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("saved preprocessing object")
            
            save_object(
                self.data_transformation_config.preprocessor_path,
                preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path
            )
            
        except Exception as e:
            raise customException(e, sys)
        
