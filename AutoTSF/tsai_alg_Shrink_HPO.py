import math

import numpy
import numpy as np
import pandas as pd
import random
import multiprocessing as mp
import time
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from functools import partial

from tsai.data.core import TSForecasting
from tsai.data.preprocessing import TSStandardize
from tsai.tslearner import TSForecaster

# algorithm list
tsai_algs = [
    'InceptionTimePlus62x62',
    'InceptionTimeXLPlus',
    'MultiInceptionTimePlus',
    'LSTMPlus',
    'MultiTSTPlus',
    'XCMPlus',
    'mWDN']
# hyper-parameter
coincidence = 3
init_scr = 90.0
shrink_num = 3
n = 3
m = 200
# init
np.set_printoptions(suppress=True, linewidth=200)
tsai_alg_param = {
    'lr': numpy.arange(1e-4, 0.1, 5e-4),
    'epoh': range(10, 100, 5),
    'bs': [8, 16, 32, 64, 128, 256, 512],
}
arr = mp.Array('d', np.zeros(n * m * 3, dtype='float'), lock=False)
tmp_arr = np.frombuffer(arr, dtype='float').reshape(n, m, 3)
for i in range(n):
    for j in range(m):
        tmp_arr[i][j][0] = init_scr
lock = mp.Lock()
lock_iter = mp.Lock()
iter_rec_arr = mp.Array('d', np.zeros(50 * 3, dtype='float'), lock=False)
t_start = time.perf_counter()


def shrink(x):
    sm = 0
    mn = 100
    l = len(tsai_alg_param[x])
    w = np.zeros(l)
    with lock:
        ar = np.frombuffer(arr, dtype='float').reshape(n, m, 3)
        for i in range(l):
            mn = min(mn, ar[x][i][0])
        mn = math.floor(mn) - shrink_num
        for i in range(l):
            if ar[x][i][0] == 100 and ar[x][i][2] > coincidence:
                w[i] = 0
            else:
                w[i] = ar[x][i][0] - mn
                sm += w[i]
    sm = sm * random.randint(1, 100) / 100
    pre = 0
    ret = 0
    for i in range(l):
        pre += w[i]
        if pre >= sm:
            ret = i
            break
    # print(x, w, sm, ret, mn)
    return ret


def select_param():
    ret_prm = np.zeros(n, dtype=int)
    for i in range(n):
        ret_prm[i] = shrink(i)
    return ret_prm


def test_model(prm_id, alg, y_train, y_test):
    truc = 0
    train_len, test_len = len(y_train), len(y_test)
    splits = [list(range(train_len if not truc else truc)),
              list(range(train_len, train_len + (test_len if not truc else truc)))]
    tfms = [None, TSForecasting()]
    batch_tfms = TSStandardize()
    tsai_model = TSForecaster(y_train, y_test, splits=splits, tfms=tfms,
                              batch_tfms=batch_tfms, bs=int(tsai_alg_param[2][prm_id[2]]), arch=alg,
                              metrics=lambda x, y: np.average(list(
                                  mean_absolute_percentage_error(xi, yi, symmetric=True) for xi, yi in
                                  zip(x.cpu().permute(0, 2, 1), y.cpu().permute(0, 2, 1)))
                              ))
    tsai_model.fit(int(tsai_alg_param[1][prm_id[1]]), lr=float(tsai_alg_param[0][prm_id[0]]))
    scr = tsai_model.final_record[-1]
    with lock:
        ar = np.frombuffer(arr, dtype='float').reshape(n, m, 3)
        for i in range(n):
            ar[i][prm_id[i]][1] += scr
            ar[i][prm_id[i]][2] += 1
            ar[i][prm_id[i]][0] = ar[i][prm_id[i]][1] / ar[i][prm_id[i]][2]
    return scr


# def test_model(prm_id, y_train, y_test, fh):
#     arima_model = ARIMA(order=(arima_param[0][prm_id[0]], arima_param[1][prm_id[1]], arima_param[2][prm_id[2]]),
#                         maxiter=arima_param[3][prm_id[3]])
#     arima_model.fit(pd.DataFrame(y_train))
#     fh_ = ForecastingHorizon(range(1, fh + 1), is_relative=True)
#     y_predict = arima_model.predict(fh_)
#     mape = mean_absolute_percentage_error(y_predict, y_test, symmetric=True)
#     scr = round(mape * 100, 2)
#     return scr


def best_param(alg, y_train, y_test):
    bst_prm = list(np.zeros(n, dtype=int))
    with lock:
        ar = np.frombuffer(arr, dtype='float').reshape(n, m, 3)
        for i in range(n):
            k = 0
            mx = 0
            for j in range(len(tsai_alg_param[i])):
                if ar[i][j][0] > mx:
                    if ar[i][j][0] == 100 and ar[i][j][2] > coincidence:
                        continue
                    mx = ar[i][j][0]
                    k = j
            bst_prm[i] = k
    return test_model(bst_prm, alg, y_train, y_test), bst_prm


def solve(x, alg, y_train, y_test):
    tmp_prm = select_param()
    train_scr = test_model(tmp_prm, alg, y_train, y_test)
    if x % 10 == 0:
        with lock_iter:
            tmp_iter_arr = np.frombuffer(iter_rec_arr, dtype='float').reshape(200, 3)
            i = int(x / 10)
            tmp_iter_arr[i][0] = x
            tmp_iter_arr[i][1] = time.perf_counter() - t_start
            tmp_iter_arr[i][2] = train_scr
            # print(tmp_iter_arr)
        # iter_rec_arr.append([x, time.perf_counter() - t_start, train_scr])
        print(str(x) + " is done.\n")


def init(ar, ite_ar, l=None, l_ite=None):
    globals()['arr'] = np.frombuffer(ar, dtype='float').reshape(n, m, 3)
    globals()['iter_rec_arr'] = np.frombuffer(ite_ar, dtype='float').reshape(50, 3)
    globals()['lock'] = l
    globals()['lock_iter'] = l_ite


def shrink_tsai(alg, y_train, y_test, rd, njob):
    pl = mp.Pool(njob, initializer=init, initargs=(arr, iter_rec_arr, lock, lock_iter))
    pl.map(partial(solve, y_train=y_train), range(rd))
    test_scr, param = best_param(alg, y_train, y_test)
    tmp_iter_arr = np.frombuffer(iter_rec_arr, dtype='float').reshape(100, 3)
    return test_scr, param
