import pandas as pd
import numpy as np
import math 

'''
This script will write a class in which we store functions for the enesmble calculations based on the other scripts model classes which should be calculate the input values for the ensemble calculations
'''



class ensemble_wrapper(ForecastingModels):
    
    def __init__(self, model_predictors):
        """
        Initialize the EnsembleModel with a list of model predictors.
        Each model predictor is assumed to be a class instance that has a method 'predict'
        returning a dataframe with columns 'date', 'actual', 'forecast', 'accuracy'.
        """
        self.model_predictors = model_predictors

    def merge_predictions(self):
        """
        Merge predictions from all models based on the 'date' column.
        """
        merged = None
        for model in self.model_predictors:
            predictions = model.predict()

            if merged is None:
                merged = predictions
            else:
                merged = merged.merge(predictions, on='date', suffixes=('', f'_{type(model).__name__}'))

        return merged

    def calculate_ensemble_forecast(self, merged_predictions):
        """
        Calculate the ensemble forecast by averaging the forecast values of all models.
        """
        forecast_columns = [col for col in merged_predictions.columns if 'forecast' in col]
        merged_predictions['ensemble_forecast'] = merged_predictions[forecast_columns].mean(axis=1)
        return merged_predictions

    def evaluate_ensemble(self, merged_predictions):
        """
        Evaluate the ensemble model by comparing the ensemble forecast with the actual values.
        """
        merged_predictions['ensemble_accuracy'] = 100 - (abs(merged_predictions['actual'] - merged_predictions['ensemble_forecast']) / merged_predictions['actual']) * 100
        return merged_predictions[['date', 'actual', 'ensemble_forecast', 'ensemble_accuracy']]

    def run_ensemble(self):
        """
        Run the ensemble model and return the final dataframe with ensemble predictions.
        """
        merged_predictions = self.merge_predictions()
        merged_predictions = self.calculate_ensemble_forecast(merged_predictions)
        final_predictions = self.evaluate_ensemble(merged_predictions)
        return final_predictions
