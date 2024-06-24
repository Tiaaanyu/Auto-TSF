from numpy import Inf
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing  # Holt-Winters
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.compose import BaggingForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import EnsembleForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from tsai.basics import *
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import numpy as np
from tsai.basics import *

import os
import logging
import logging.handlers
from datetime import datetime
from generateFeatures import get_meta_data


# Step 2: Configure the logging settings
def setup_logging():
    # Create a log directory if it doesn't exist
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_level = logging.INFO

    # Set the format of log messages
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Get the current timestamp as a string
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Specify the log file path with a unique name using the timestamp
    log_file = os.path.join(log_directory, f'my_machine_learning_app_{timestamp}.log')

    # Set up a rotating file handler for log files (to limit log size)
    max_log_size_bytes = 1000000  # 1 MB
    backup_log_count = 3  # Number of backup log files to keep
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_log_size_bytes,
                                                        backupCount=backup_log_count)

    # Create a formatter and attach it to the file handler
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    # Create a logger and set the logging level
    logger = logging.getLogger('my_machine_learning_app')
    logger.setLevel(log_level)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


# Call the function to set up the logger
logger = setup_logging()

dataset_names = [
    # "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    # 'm4_yearly_dataset',
    'm4_quarterly_dataset',
    'm4_monthly_dataset',
    'm4_weekly_dataset',
    'm4_daily_dataset',
    'm4_hourly_dataset',
    "nn5_weekly_dataset",
    "nn5_daily_dataset_without_missing_values",
    'electricity_hourly_dataset',
    'electricity_weekly_dataset'
    'tourism_yearly_dataset',
    'tourism_quarterly_dataset',
    'tourism_monthly_dataset'
]


def get_dataset(datasetname):
    if "ETT" in datasetname:
        ts = get_long_term_forecasting_data(datasetname)
        ts = ts.values[:, 1:].astype(float)
        scaler = StandardScaler()
        scaler.fit(ts)
        ts = scaler.transform(ts)
        train_, _, test_, _ = train_test_split(ts, ts, test_size=0.2, shuffle=False)
        train_sd = SlidingWindow(100 - 8, horizon=8)
        test_sd = SlidingWindow(100 - 8, horizon=8, stride=100)
        train = train_sd(train_)
        test = test_sd(test_)
        test_cat = test_
        return train, test, 8, test_cat
    if any(dname in datasetname for dname in ['m4', 'nn5', 'tourism', 'electricity']):
        ts = get_Monash_forecasting_data(datasetname)
        time_series_names = ts.series_name.unique()
        ts_data = ts.values[:, 2]
        scaler = StandardScaler()
        print("transforming ...")
        scaler.fit(ts_data[None])
        ts.values[:, 2] = scaler.transform(ts_data[None])[0]
        sample_datasets = [ts[ts['series_name'] == tsn].values[:, 2:] for tsn in time_series_names]
        train_, _, test_, _ = train_test_split(sample_datasets, sample_datasets, test_size=0.2, shuffle=False)
        test_cat = np.concatenate(test_, axis=0)
        min_len = min([len(sd) for sd in sample_datasets] + [100])
        fh = min(8, min_len // 3)
        print("sliding ...")
        sd = SlidingWindow(min_len - fh, horizon=fh, stride=min(min_len, len(ts) // 20000 + 1))
        print("stride: ", min(min_len, len(ts) // 20000 + 1))
        print("min_len: ", min_len)
        test_sd = SlidingWindow(min_len - fh, horizon=fh, stride=min_len)

        def adapt_dim(tup):
            a, b = tup
            if len(a.shape) == 2:
                a = a[:, None]
            if len(b.shape) == 2:
                b = b[:, None]
            # print(a.shape, b.shape)
            return (a, b)

        train = [np.concatenate(item, axis=0).astype(float)
                 for item in list(zip(*[adapt_dim(sd(x)) for x in train_]))]
        test = [np.concatenate(item, axis=0).astype(float)
                for item in zip(*[adapt_dim(test_sd(x[-min_len:])) for x in test_])]

        return train, test, fh, test_cat


forecasters = [
    ("trend", PolynomialTrendForecaster()),
    ("naive", NaiveForecaster())
]

convert = {
    "exp": "ExponentialSmoothing",
    "ari": "ARIMA",
    "sari": "SARIMAX",
    "a-ets": "AutoETS",
    "bag": "BaggingForecaster",
    "tre": "TrendForecaster",
    "poly": "PolynomialTrendForecaster",
    "stl": "STLForecaster",
    "pro": "Prophet"
}
algs = {
    "ExponentialSmoothing": ExponentialSmoothing,
    "ARIMA": ARIMA,
    "SARIMAX": SARIMAX,
    "AutoETS": AutoETS,
    "BaggingForecaster": BaggingForecaster,
    "TrendForecaster": TrendForecaster,
    "PolynomialTrendForecaster": PolynomialTrendForecaster,
    "STLForecaster": STLForecaster,
    "Prophet": Prophet,
}
algs_names = list(algs.keys())


def test_sktime_method(alg, test, fh, alg_args=()):
    fh_ = ForecastingHorizon(range(1, fh + 1), is_relative=True)
    train_y = test[0]
    test_y = test[1]
    len_test = len(train_y)
    mapes = []
    for i in tqdm(range(len_test)):
        train_y_ = train_y[i].transpose()
        test_y_ = test_y[i].transpose()
        alg_ = algs[alg](*alg_args)
        alg_.fit(pd.DataFrame(train_y_))
        pred_y = alg_.predict(fh_)
        mape_ = mean_absolute_percentage_error(pred_y, test_y_, symmetric=True)
        mapes.append(mape_)
    mape = float(np.mean(mapes))
    return mape


tsai_algs = [
    'InceptionTimePlus62x62',
    'InceptionTimeXLPlus',
    'MultiInceptionTimePlus',
    'LSTMPlus',
    'MultiTSTPlus',
    'XCMPlus',
    'mWDN']


def adapt(x, name="X"):
    print(name + ": ")
    print(x.shape)
    print(x)
    return x


def test_tsai_methods(alg, train, test, epoch=10, lr=1e-3, truc=0):
    train_len, test_len = len(train[0]), len(test[0])
    splits = [list(range(train_len if not truc else truc)),
              list(range(train_len, train_len + (test_len if not truc else truc)))]
    # print(train[0].shape, train[1].shape, test[0].shape, test[1].shape)
    X, y = np.concatenate([train[0], test[0]], axis=0), \
        np.concatenate([train[1], test[1]], axis=0)
    print(X.shape, y.shape)
    tfms = [None, TSForecasting()]
    batch_tfms = TSStandardize()
    fcst = TSForecaster(X, y, splits=splits, path='models', tfms=tfms,
                        batch_tfms=batch_tfms, bs=512, arch=alg, metrics=lambda x, y: np.average(list(
            mean_absolute_percentage_error(xi, yi, symmetric=True) for xi, yi in
            zip(x.cpu().permute(0, 2, 1), y.cpu().permute(0, 2, 1)))
        ))
    fcst.fit_one_cycle(epoch, lr)
    return fcst.final_record[-1]


def calc_epoch(batchs):
    return 6918000 // batchs + 1


performance = []
valid_data = []
for data in dataset_names:
    try:
        train_, test_, fh, _ = get_dataset(data)
        valid_data.append(data)
    except:
        continue
    alg_performance = []
    epochs = calc_epoch(train_[0].shape[0])
    for n in tsai_algs:
        try:
            mape_ = test_tsai_methods(n, train_, test_, epochs)
        except:
            mape_ = Inf
            logger.error("An error occurred:", exc_info=True)
        logger.info("alg: " + n + ", data: " + data + ", smape: " + str(mape_))
        alg_performance.append(mape_)
    for n in algs_names:
        try:
            mape_ = test_sktime_method(n, test_, fh)
        except:
            mape_ = Inf
            logger.error("An error occurred:", exc_info=True)
        alg_performance.append(mape_)
        logger.info("alg: " + n + ", data: " + data + ", smape: " + str(mape_))

    performance.append(alg_performance)

logger.info(str(performance))

col = tsai_algs + algs_names
row = dataset_names
logger.info("col: " + str(col))
logger.info('row: ' + str(row))
performance = np.array(performance)
# performance.columns = col
# performance.index = row

meta_data = get_meta_data(dataset_names)
features = []
for d in dataset_names:
    features.append(meta_data[d])
features = np.array(features)

y = np.argmin(performance, axis=1)
best_performance = np.max(performance, axis=1)

classifier = RandomForestClassifier()
classifier.fit(features, y)
