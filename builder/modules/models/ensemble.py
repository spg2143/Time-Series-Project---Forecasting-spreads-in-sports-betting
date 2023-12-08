import pandas as pd
import numpy as np
import math 

'''
This script will write a class in which we store functions for the enesmble calculations based on the other scripts model classes which should be calculate the input values for the ensemble calculations
'''



class ensemble_wrapper(Arimax, Var, Xgboost):
    
    def __init__(self):
        self.prediction_frame: pd.DataFrame
        self.