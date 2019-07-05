import numpy as np

from scipy import stats

def rss(Y1, Y2):
    '''
    ISLR pg. 72

    :param Y1:
    :param Y2:
    :return:
    '''
    return np.power( Y1 - Y2, 2).sum()

def rse(rss, n, p):
    '''
    ISLR pg. 80

    :param rss:
    :param n:
    :param p:
    :return:
    '''
    return np.sqrt( rss / (n - p - 1))

def rse_arr(Y1, Y2, p):
    '''

    :param Y1:
    :param Y2:
    :param p:
    :return:
    '''
    return rse(rss(Y1, Y2), Y1.size, p)

def standard_error(var, X):
    '''
    ISLR pg. 66

    :param var:
    :param X:
    :return:
    '''
    x = var / np.power(X - X.mean(), 2).sum()
    return np.sqrt(x)

def t_stat(B, standard_error):
    '''
    ISLR pg. 67

    :param B:
    :return:
    '''
    return B / standard_error

def two_sided_p_value(t_stat, n):
    '''

    :param t_stat:
    :return:
    '''
    return stats.t.sf(np.abs(t_stat), n - 1) * 2

def tss(Y):
    '''
    ISLR pg. 76

    :param Y:
    :return:
    '''
    return np.power(Y - Y.mean(), 2).sum()

def f_stat(tss, rss, n, p):
    '''
    ISLR pg. 75

    :param tss:
    :param rss:
    :param n:
    :param p:
    :return:
    '''
    return ((tss - rss) / p) / (rss / (n - p - 1))




