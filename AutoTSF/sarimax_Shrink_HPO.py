import math
from functools import partial

import numpy as np
import pandas as pd
import random
import multiprocessing as mp
import time
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# hyper-parameter
coincidence = 3
init_scr = 90.0
shrink_num = 3
n = 4
m = 15
# init
np.set_printoptions(suppress=True, linewidth=200)
sarimax_param = {
    'p': range(0, 11, 1),
    'd': range(0, 6, 1),
    'q': range(0, 6, 1),
    'trend': ['n', 'c', 't', 'ct']
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
    l = len(sarimax_param[x])
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


def train_model(prm_id, y_train, fh):
    sarimax_model = SARIMAX(
        order=(sarimax_param[0][prm_id[0]], sarimax_param[1][prm_id[1]], sarimax_param[2][prm_id[2]]),
        trend=sarimax_param[3][prm_id[3]])
    sarimax_model.fit(pd.DataFrame(y_train))
    fh_ = ForecastingHorizon(range(1, fh + 1), is_relative=True)
    y_predict = sarimax_model.predict(fh_)
    mape = mean_absolute_percentage_error(y_predict, y_train, symmetric=True)
    scr = round(mape * 100, 2)
    with lock:
        ar = np.frombuffer(arr, dtype='float').reshape(n, m, 3)
        for i in range(n):
            ar[i][prm_id[i]][1] += scr
            ar[i][prm_id[i]][2] += 1
            ar[i][prm_id[i]][0] = ar[i][prm_id[i]][1] / ar[i][prm_id[i]][2]
        # print(f"{mp.current_process().name}\n{ar[:]}")
    return scr


def test_model(prm_id, y_train, y_test, fh):
    sarimax_model = SARIMAX(
        order=(sarimax_param[0][prm_id[0]], sarimax_param[1][prm_id[1]], sarimax_param[2][prm_id[2]]),
        trend=sarimax_param[3][prm_id[3]])
    sarimax_model.fit(pd.DataFrame(y_train))
    fh_ = ForecastingHorizon(range(1, fh + 1), is_relative=True)
    y_predict = sarimax_model.predict(fh_)
    mape = mean_absolute_percentage_error(y_predict, y_test, symmetric=True)
    scr = round(mape * 100, 2)
    return scr


def best_param(y_train, y_test, fh):
    bst_prm = list(np.zeros(n, dtype=int))
    with lock:
        ar = np.frombuffer(arr, dtype='float').reshape(n, m, 3)
        for i in range(n):
            k = 0
            mx = 0
            for j in range(len(sarimax_param[i])):
                if ar[i][j][0] > mx:
                    if ar[i][j][0] == 100 and ar[i][j][2] > coincidence:
                        continue
                    mx = ar[i][j][0]
                    k = j
            bst_prm[i] = k
    return test_model(bst_prm, y_train, y_test, fh), bst_prm


def solve(x, y_train, fh):
    tmp_prm = select_param()
    train_scr = train_model(tmp_prm, y_train, fh)
    if x % 10 == 0:
        with lock_iter:
            tmp_iter_arr = np.frombuffer(iter_rec_arr, dtype='float').reshape(50, 3)
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


def shrink_sarimax(y_train, y_test, fh, rd, njob):
    pl = mp.Pool(njob, initializer=init, initargs=(arr, iter_rec_arr, lock, lock_iter))
    pl.map(partial(solve, y_train=y_train), range(rd))
    test_scr, param = best_param(y_train, y_test, fh)
    train_scr = train_model(param, y_train, fh)
    # ar = np.frombuffer(arr, dtype='float').reshape(n, m, 3)
    tmp_iter_arr = np.frombuffer(iter_rec_arr, dtype='float').reshape(50, 3)
    # print(ar, '\n', train_scr, test_scr, param)
    # print(ar, tmp_iter_arr)
    return test_scr, param
