import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy import stats

# TODO: "Overload" functions so that I can not redo work when calling t-value, p-values, etc. - https://stackoverflow.com/questions/7113032/overloaded-functions-in-python
# TODO: Add plots.  I believe I have some of the plots to add on my hand written notes.
# TODO: Add mixed selection automation (ISLR pg. 79). POTENTIAL_OPTIONS: Fix skew response, Fix skew predictors, lambda new columns, collinearity drop, collinearity combine

class LrMetrics:
    def __init__(self, X, Y, reg=None):
        # If reg = None then run create a scikit learn linear model
        if reg == None:
            reg = LinearRegression()
            reg.fit(X, Y)

        self.reg = reg;

        # Get sub-fields from data
        self.n = Y.size;
        self.features = X.columns;
        self.p = X.columns.size

        # Set X and Y to numpy arrays
        self.X = np.concatenate([np.ones(Y.size).reshape(Y.size, 1), np.array(X)], axis=1)
        self.Y = np.array(pd.DataFrame(Y))

        # Create B where the first value is the intercept and the following values are the coefficients
        self.B = np.append(reg.intercept_, reg.coef_).reshape(self.X.shape[1], 1)

        # Set fields to None
        self.rss = None
        self.rse = None


class MixedSelection:
    def __int__(self, X, Y):
        self.X = X;
        self.Y = Y




    def simple_linear_regressions(self, sort=True):
        simp_lrs = list(None * self.Y.size)

        for feat in self.X.columns:



        # If sort=True then sort values in list
        if sort:
            pass

        return -99




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

    # Get general information
    summary['general'] = __general(reg, X, Y)

    return summary

# TODO: Add p-value and adjusted R^2 to __general.  Currently they have placeholders of -99.
def __general(reg, X, Y):
    '''

    :param reg:
    :param X:
    :param Y:
    :return:
    '''
    idx = ['rse', 'f-stat', 'p-value', 'R^2', 'adj_R^2']

    values = list()
    values.append(__get_rse(reg, X, Y))
    values.append(__f_stat(reg, X, Y))
    values.append(-99)
    values.append(reg.score(X, Y))
    values.append(-99)

    se = pd.Series(data=values, index=idx)
    se.name = 'General'
    return se


def __f_stat(reg, X, Y):
    '''
    ISLR pg. 75

    :param reg:
    :param X:
    :param Y:
    :return:
    '''
    n = Y.size
    p = X.columns.size
    tss = np.power(Y - Y.mean(), 2).sum()
    rss = np.power(reg.predict(X) - Y, 2).sum()

    return ((tss - rss) / p) / (rss / (n - p - 1))


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



'''
Residuals:
    Min      1Q  Median      3Q     Max 
-8.8277 -0.8908  0.2418  1.1893  2.8292 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  2.938889   0.311908   9.422   <2e-16 ***
TV           0.045765   0.001395  32.809   <2e-16 ***
radio        0.188530   0.008611  21.893   <2e-16 ***
newspaper   -0.001037   0.005871  -0.177     0.86    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 1.686 on 196 degrees of freedom
Multiple R-squared:  0.8972,	Adjusted R-squared:  0.8956 
F-statistic: 570.3 on 3 and 196 DF,  p-value: < 2.2e-16
'''

