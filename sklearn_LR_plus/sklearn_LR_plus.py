import numpy as np
import pandas as pd

def get_standard_errors(reg, X, Y):
    '''
    http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis

    :param reg:
    :param X:
    :param Y:
    :return:
    '''
    # X_arr = pd.concat( [pd.DataFrame({'intercept' : np.ones(Y.size)}), X], axis=1, ignore_index=True )
    X_arr = np.concatenate([np.ones(Y.size).reshape(Y.size, 1), np.array(X)], axis=1)
    # X_arr = np.array(X_arr)
    Y_arr = np.array(pd.DataFrame(Y))

    B = np.append(reg.intercept_, reg.coef_).reshape(X_arr.shape[1], 1)

    rse = np.sqrt(np.power((X_arr @ B) - Y_arr, 2).sum() / (Y_arr.size - X_arr.shape[1]))
    var = rse**2

    C = var * np.linalg.inv(X_arr.T @ X_arr)
    return pd.Series(data=np.sqrt(np.diag(C)), index=(['Intercept'] + X.columns.tolist()))


def get_t_values(reg, X, Y):
    '''
    ISLR pg. 72?

    :param reg:
    :param X:
    :param y:
    :return:
    '''
    B = np.append(reg.intercept_, reg.coef_)
    rse = __get_rse(reg, X, Y)

    return pd.Series(data=(B / rse), index=['Intercept'] + X.columns.tolist())


def __get_rse(reg, X, Y):
    '''
    ISLR pg. 68?

    :param reg:
    :param X:
    :param Y:
    :return:
    '''
    rss = np.power( reg.predict(X) - Y, 2).sum()
    return np.sqrt( rss / (Y.size - X.columns.size - 1))



