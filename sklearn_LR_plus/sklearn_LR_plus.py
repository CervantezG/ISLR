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
        self.features = X.columns.tolist();
        self.p = X.columns.size

        # Set X and Y to numpy arrays
        self.X = np.concatenate([np.ones(Y.size).reshape(Y.size, 1), np.array(X)], axis=1)
        self.Y = Y.values

        # Create B where the first value is the intercept and the following values are the coefficients
        self.B = np.append(self.reg.intercept_, self.reg.coef_)

        # Set fields to None
        self.rss = None
        self.coef_rse = None
        self.standard_errors = None
        self.t_values = None
        self.rse = None
        self.tss = None
        self.f_stat = None


    def get_coef_rse(self):
        if self.coef_rse is None:
            self.coef_rse = np.sqrt(
                np.power((self.X @ self.B.reshape(self.X.shape[1], 1)) - np.array(pd.DataFrame(self.Y)), 2).sum() / (
                            self.Y.size - self.X.shape[1]))
        return self.coef_rse

    def get_standard_errors(self):
        '''
        http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
    
        :param reg:
        :param X:
        :param Y:
        :return:
        '''
        if self.standard_errors is None:
            var = self.get_coef_rse()**2

            C = var * np.linalg.inv(self.X.T @ self.X)
            self.standard_errors = pd.Series(data=np.sqrt(np.diag(C)), index=(['Intercept'] + self.features))
        return self.standard_errors

    def get_t_values(self):
        '''
        ISLR pg. 72?

        :param reg:
        :param X:
        :param y:
        :return:
        '''
        if self.t_values is None:
            self.t_values = self.B / self.get_standard_errors()
        return self.t_values

    def get_p_values(self):
        '''
        ISLR pg. 72?

        :param reg:
        :param X:
        :param y:
        :return:
        '''
        # t_values = get_t_values(reg, X, Y)
        p_func = lambda t: stats.t.sf(np.abs(t), self.n - 1) * 2

        return self.get_t_values().apply(p_func)

    def get_coef_tests(self):
        columns = ['Coef.', 'Std. Error', 't-value', 'p-value']
        idx = ['Intercept'] + self.features
        se_B = pd.Series(self.B, index=idx)
        s_errors = self.get_standard_errors()
        t_values = self.get_t_values()
        p_values = self.get_p_values()

        coef_df = pd.concat([se_B, s_errors, t_values, p_values], axis=1, ignore_index=True)
        coef_df.columns = columns
        coef_df.index = idx

        return coef_df

    def get_residuals_data(self):
        residuals = pd.Series(self.Y - self.reg.predict(self.X[:, 1:])).describe()
        residuals.name = 'Residuals'
        return residuals.drop(labels=['count', 'mean'])

    # TODO: Add p-value and adjusted R^2 to __general.  Currently they have placeholders of -99.
    def get_general_data(self):
        '''

        :param reg:
        :param X:
        :param Y:
        :return:
        '''
        idx = ['rse', 'f-stat', 'p-value', 'R^2', 'adj_R^2']

        values = list()
        values.append(self.get_rse())
        values.append(self.get_f_stat())
        values.append(-99)
        values.append(self.reg.score(self.X[:, 1:], self.Y))
        values.append(-99)

        se = pd.Series(data=values, index=idx)
        se.name = 'General'
        return se


    def get_tss(self):
        if self.tss is None:
            self.tss = np.power(self.Y - self.Y.mean(), 2).sum()
        return self.tss


    def get_rss(self):
        if self.rss is None:
            self.rss = np.power(self.reg.predict(self.X[:, 1:]) - self.Y, 2).sum()
        return self.rss


    def get_rse(self):
        if self.rse is None:
            self.rse = np.sqrt(self.get_rss() / (self.Y.size - self.p - 1))
        return self.rse


    def get_f_stat(self):
        if self.f_stat is None:
            self.f_stat = ((self.get_tss() - self.get_rss()) / self.p) / (self.get_rss() / (self.n - self.p - 1))
        return self.f_stat


    # tss = np.power(Y - Y.mean(), 2).sum()
    # rss = np.power(reg.predict(X) - Y, 2).sum()
    #
    # return ((tss - rss) / p) / (rss / (n - p - 1))



############################################################################



class MixedSelection:
    def __int__(self, X, Y):
        self.X = X;
        self.Y = Y


    def simple_linear_regressions(self, sort=True):
        simp_lrs = list(None * self.Y.size)

        # Create simple linear regressions for all columns in self.X
        for feat in self.X.columns:
            simp_lrs.append(LrMetrics(self.X[feat], self.Y))

        # If sort=True then sort values in list

        if sort:
            rss_f = lambda x: x.rss
            simp_lrs.sort(key=rss_f)

        return simp_lrs


######################################################################


# def summary(reg, X, Y):
#     summary = dict()
#
#     # Get information on hypothesis tests ran against individual coefficients
#     summary['coef_tests'] = __coef_tests(reg, X, Y)
#
#     # Get residuals distribution information
#     summary['residuals'] = __residuals_data(reg, X, Y)
#
#     # Get general information
#     summary['general'] = __general(reg, X, Y)
#
#     return summary

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

