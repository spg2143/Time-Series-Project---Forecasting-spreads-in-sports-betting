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

    def calculate_weights(self, merged_predictions):
        """
        Calculate weights for each model inversely proportional to their mean accuracy.
        """
        weight_columns = {}
        for col in merged_predictions.columns:
            if 'accuracy' in col:
                model_name = col.split('_')[0]
                mean_accuracy = merged_predictions[col].mean()
                weight_columns[model_name] = 1 / mean_accuracy if mean_accuracy != 0 else 0

        total_weight = sum(weight_columns.values())
        normalized_weights = {k: v / total_weight for k, v in weight_columns.items()}

        return normalized_weights

    def calculate_weighted_ensemble_forecast(self, merged_predictions, weights):
        """
        Calculate the weighted ensemble forecast.
        """
        merged_predictions['ensemble_forecast'] = 0
        for model, weight in weights.items():
            forecast_col = f'{model}_forecast'
            merged_predictions['ensemble_forecast'] += weight * merged_predictions[forecast_col]

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
        weights = self.calculate_weights(merged_predictions)
        merged_predictions = self.calculate_weighted_ensemble_forecast(merged_predictions, weights)
        final_predictions = self.evaluate_ensemble(merged_predictions)
        return final_predictions