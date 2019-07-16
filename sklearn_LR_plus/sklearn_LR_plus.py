import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy import stats

# TODO: Add plots.  I believe I have some of the plots to add on my hand written notes.

class LrMetrics:
    '''A class for linear regression metrics.  These metrics are inspired by ISLR Ch. 3 and R.

    TODO: Add attributes and change to use properties.
    '''


    def __init__(self, X, Y, reg=None):
        '''A LrMetric constructor that takes a X, Y, and an optional linear model.  If the linear model is not provided
        one is created.

        :param X: A pandas DataFrame of data used for predictions
        :param Y: A pandas Series of the value trying to be predicted
        :param reg: A linear regression estimator
        '''
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
        self._rss = None
        self.coef_rse = None
        self.standard_errors = None
        self.t_values = None
        self.rse = None
        self.tss = None
        self.f_stat = None


    def get_coef_rse(self):
        '''Calculates the residual standard errors for all coefficients.

        :return: a numpy array of the RSE of the coefficients
        '''
        if self.coef_rse is None:
            self.coef_rse = np.sqrt(
                np.power((self.X @ self.B.reshape(self.X.shape[1], 1)) - np.array(pd.DataFrame(self.Y)), 2).sum() / (
                            self.Y.size - self.X.shape[1]))
        return self.coef_rse

    def get_standard_errors(self):
        '''Calculates the standard errors of the model.  For more information on implementation see link below.
        http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis

        :return: A Series of standard errors
        '''
        if self.standard_errors is None:
            var = self.get_coef_rse()**2

            C = var * np.linalg.inv(self.X.T @ self.X)
            self.standard_errors = pd.Series(data=np.sqrt(np.diag(C)), index=(['Intercept'] + self.features))
        return self.standard_errors

    def get_t_values(self):
        '''Calculates the t-values for the coefficients.  See more about t-values in  ISLR pg. 72.

        :return: A Series of t-values
        '''
        if self.t_values is None:
            self.t_values = self.B / self.get_standard_errors()
        return self.t_values

    def get_p_values(self):
        '''Calculates the p-values for the coefficients.  See more about p-values in  ISLR pg. 72.

        :return: A Series of p-values
        '''
        # t_values = get_t_values(reg, X, Y)
        p_func = lambda t: stats.t.sf(np.abs(t), self.n - 1) * 2

        return self.get_t_values().apply(p_func)

    def get_coef_tests(self):
        '''Get Coefficient, Standard Error, t-value, and p-value in the form of a DataFrame.  This is modeling on the
        summary function in R.

        :return: DataFrame of Coefficient, Standard Error, t-value, and p-value
        '''
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
        '''Get the residual standard error, the f-statistic, p-value, R^2, and adjusted R^2 in the form on a
        Series.  This is inspired by the summary function in R.

        :return: The RSE, the f-stat, p-value, R^2, and adjusted R^2 in the form on a Series
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
        '''Gets the total sum of squares.  A calculation will happen if the first time this method is being called.

        :return: the total sum of squares
        '''
        if self.tss is None:
            self.tss = np.power(self.Y - self.Y.mean(), 2).sum()
        return self.tss

    def get_rss(self):
        '''Gets the residual sum of squares.  A calculation will happen if the first time this method is being called.

        :return: the residual sum of squares
        '''
        if self._rss is None:
            self._rss = np.power(self.reg.predict(self.X[:, 1:]) - self.Y, 2).sum()
        return self._rss

    def get_rse(self):
        '''Gets the residual standard error.  A calculation will happen if the first time this method is being called.

        :return: the residual standard error
        '''
        if self.rse is None:
            self.rse = np.sqrt(self.get_rss() / (self.Y.size - self.p - 1))
        return self.rse

    def get_f_stat(self):
        '''Gets the f-statistic.  A calculation will happen if the first time this method is being called.

        :return: the f-statistic
        '''
        if self.f_stat is None:
            self.f_stat = ((self.get_tss() - self.get_rss()) / self.p) / (self.get_rss() / (self.n - self.p - 1))
        return self.f_stat

    def get_high_p_features(self, q=0.05):
        '''A function to get the p-values where the the value is greater than q.

        :param q: p-value threshold
        :return: list of features with a high p-value
        '''
        return self.get_p_values().index[self.get_p_values() > q].tolist()


class MixedSelection:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


    def simple_linear_regressions(self, sort=True):
        simp_lrs = list()

        # Create simple linear regressions for all columns in self.X
        for feat in self.X.columns:
            simp_lrs.append(LrMetrics(pd.DataFrame(self.X[feat]), self.Y))

        # If sort=True then sort values in list
        if sort:
            rss_f = lambda x: x.get_rss()
            simp_lrs.sort(key=rss_f)

        return simp_lrs

    def mixed_selection(self, q=0.05):
        simp_LRs = self.simple_linear_regressions()

        features = list()

        for i in range(len(simp_LRs)):
            if len(simp_LRs[i].features) > 1:
                raise RuntimeError(
                    'Linear Regressions had more than a single feature when they were supposed to be simple.')
            feature = simp_LRs[i].features[0]
            features.append(feature)

            self.__check_features_reg(features, q)

        final_X = self.X[features]
        return LrMetrics(final_X, self.Y)


    def __check_features_reg(self, features, q):
        '''
        Create a linear regression with all features and then remove features that have above q value.

        :param features: the features of the linear regression
        :param q: p-value threshold
        '''
        temp_X = self.X[features]
        multi_reg = LrMetrics(temp_X, self.Y)
        for feat in multi_reg.get_high_p_features(q):
            if feat != 'Intercept':
                features.remove(feat)
                self.__check_features_reg(features, q)


######################################################################








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

