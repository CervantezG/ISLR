import numpy as np
import pandas as pd

from scipy import stats

# TODO: "Overload" functions so that I can not redo work when calling t-value, p-values, etc. - https://stackoverflow.com/questions/7113032/overloaded-functions-in-python

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
    standard_errors = get_standard_errors(reg, X, Y)

    return B / standard_errors


def get_p_values(reg, X, Y):
    '''
    ISLR pg. 72?

    :param reg:
    :param X:
    :param y:
    :return:
    '''
    t_values = get_t_values(reg, X, Y)
    n = Y.size
    p_func = lambda t: stats.t.sf(np.abs(t), n - 1) * 2

    return t_values.apply(p_func)


def summary(reg, X, Y):
    summary = dict()

    # Get information on hypothesis tests ran against individual coefficients
    summary['coef_tests'] = __coef_tests(reg, X, Y)

    # Get residuals distribution information
    summary['residuals'] = __residuals_data(reg, X, Y)

    return summary


def __coef_tests(reg, X, Y):
    columns = ['Coef.', 'Std. Error', 't-value', 'p-value']
    idx = ['Intercept'] + X.columns.tolist()
    B = pd.Series(np.append(reg.intercept_, reg.coef_), idx)
    s_errors = get_standard_errors(reg, X, Y)
    t_values = get_t_values(reg, X, Y)
    p_values = get_p_values(reg, X, Y)

    summ_df = pd.concat([B, s_errors, t_values, p_values], axis=1, ignore_index=True)
    summ_df.columns = columns
    summ_df.index = idx

    return summ_df

def __residuals_data(reg, X, Y):
    residuals = (Y - reg.predict(X)).describe()
    residuals.name = 'Residuals'
    return residuals.drop(labels=['count', 'mean'])


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



