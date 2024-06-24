import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from generateFeatures import GenerateFeature
import arima_Shrink_HPO
import exponential_Smoothing_Shrink_HPO
import sarimax_Shrink_HPO
import bagging_Forecaster_Shrink_HPO
import polynomial_Trend_Forecaster_Shink_HPO
import stl_Forecaster_Shrink_HPO
import prophet_Shrink_HPO
import tsai_alg_Shrink_HPO

with open('meta_learner.pkl', 'rb') as f:
    meta_learner = pickle.load(f)

tsai_algs = [
    'InceptionTimePlus62x62',
    'InceptionTimeXLPlus',
    'MultiInceptionTimePlus',
    'LSTMPlus',
    'MultiTSTPlus',
    'XCMPlus',
    'mWDN']


def alg_selection(data_path: str) -> str:
    prev_time = time.time()
    time_feature = GenerateFeature(data_path)
    meta_feature = time_feature.generate_meta_features()
    optimal_algorithm = meta_learner.predict(meta_feature)
    alg_selection_timecost = time.time() - prev_time
    print("Time cost for automatic algorithm selection is " + str(alg_selection_timecost) + '\n')
    print("The optimal algorithm is " + optimal_algorithm + '\n')
    return str(optimal_algorithm)


def hpo(data_path: str, algorithm: str) -> (float, dict):
    data = np.loadtxt(data_path, dtype=np.float32, delimiter=',')
    train, test = train_test_split(data, test_size=0.2)
    result = {}
    best_smape = 0
    best_hyperparameter = []
    if algorithm in tsai_algs:
        best_smape, best_hyperparameter = tsai_alg_Shrink_HPO.shrink_tsai(algorithm, train, test, 200, 16)
        result = {'lr': best_hyperparameter[0], 'epoh': best_hyperparameter[1], 'bs': best_hyperparameter[2]}
    elif algorithm == 'ARIMA':
        best_smape, best_hyperparameter = arima_Shrink_HPO.shrink_arima(train, test, 8, 200, 16)
        result = {'order': best_hyperparameter[:3], 'max_iter': best_hyperparameter[3:]}
    elif algorithm == 'ExponentialSmoothing':
        best_smape, best_hyperparameter = exponential_Smoothing_Shrink_HPO.shrink_exp_smoothing(train, test, 8, 200, 16)
        result = {'trend': best_hyperparameter[0], 'seasonal': best_hyperparameter[1], 'sp': best_hyperparameter[2]}
    elif algorithm == 'SARIMAX':
        best_smape, best_hyperparameter = sarimax_Shrink_HPO.shrink_sarimax(train, test, 8, 200, 16)
        result = {'order': best_hyperparameter[:3], 'trend': best_hyperparameter[3:]}
    elif algorithm == 'BaggingForecaster':
        best_smape, best_hyperparameter = bagging_Forecaster_Shrink_HPO.shrink_bagging_forecaster(train, test, 8, 200,
                                                                                                  16)
        result = {'sp': best_hyperparameter[0]}
    elif algorithm == 'PolynomialTrendForecaster':
        best_smape, best_hyperparameter = polynomial_Trend_Forecaster_Shink_HPO.shrink_polynomial_trend_forecaster(
            (train, test, 8, 200, 16))
        result = {'degree': best_hyperparameter[0]}
    elif algorithm == 'STLForecaster':
        best_smape, best_hyperparameter = stl_Forecaster_Shrink_HPO.shrink_stl_forecaster(train, test, 8, 200, 16)
        result = {'sp': int(best_hyperparameter[0]), 'seasonal': int(best_hyperparameter[1])}
    elif algorithm == 'Prophet':
        best_smape, best_hyperparameter = prophet_Shrink_HPO.shrink_prophet(train, test, 8, 200, 16)
        result = {'seasonality_mode': best_hyperparameter[0]}
    else:
        print("No hyperparameter to tune.")  # For example, Trend Forecaster and AutoETS

    return best_smape, result


if __name__ == "__main__":
    data_path = 'AirPassenger.csv'
    alg = alg_selection(data_path)
    smape, hp = hpo(data_path, alg)
    print(alg)
