#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 23:07
# @Author  : ZSH
'''

'''
import numpy as np
from scipy.linalg import toeplitz
class PACF():
    '''
    
    '''
    def __init__(self):
        pass
    def _cal_my_yule_walker(self, x,nlags=5):
        """
            
            :param x:
            :param nlags:
            :return:
            """
        x = np.array(x, dtype=np.float64)
        x -= x.mean()
        n = x.shape[0]

        r = np.zeros(shape=nlags + 1, dtype=np.float64)
        r[0] = (x ** 2).sum() / n

        for k in range(1, nlags + 1):
            r[k] = (x[0:-k] * x[k:]).sum() / (n - k * 1)
        R = toeplitz(c=r[:-1])
        result = np.linalg.solve(R, r[1:])
        return result
    def cal_my_pacf_yw(self,x, nlags=5):
        """
            
            :param x:
            :param nlags:
            :return:
            """
        pacf = np.empty(nlags + 1) * 0
        pacf[0] = 1.0
        for k in range(1, nlags + 1):
            pacf[k] = self._cal_my_yule_walker(x, nlags=k)[-1]

        return pacf