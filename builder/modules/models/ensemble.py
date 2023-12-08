import pandas as pd
import numpy as np
import math 

'''
This script will write a class in which we store functions for the enesmble calculations based on the other scripts model classes which should be calculate the input values for the ensemble calculations
'''



class ensemble_wrapper():
    
    def __init__(self):
        self.a = 1
        self.b = 2
    
    def add_ab(self):
        return self.a + self.b