import numpy as np

def rss(Y1, Y2):
    return np.power( Y1 - Y2, 2).sum()

def rse(rss, n):
    return np.sqrt(rss / (n - 2)) # May be n - p - 1 where p is degrees of freedom

def rse_arr(Y1, Y2):
    return rse( rss(Y1, Y2), Y1.size)

def standard_error(var, X):
    return np.sqrt( var / ( X - np.power(X.mean(), 2) ) )