import unittest

import numpy as np
import pandas as pd

import sklearn_LR_plus as lrp

from sklearn.linear_model import LinearRegression


class TestGet_standard_errors(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.get_standard_errors().
    '''

    def test_get_standard_errors_advertising_simple(self):
        '''
        Test the standard error on simple linear regression using Advertising data and ISLR pg. 68, 72
        '''
        standard_errors = {'TV' : np.array([0.4578, 0.0027]),
                           'radio' : np.array([0.563, 0.020]),
                           'newspaper' : np.array([0.621, 0.017])}

        places = 4

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for col in standard_errors:
            X = pd.DataFrame(df[col])

            reg = LinearRegression()
            reg.fit(X, Y)

            expected = pd.Series(data = standard_errors[col],
                                 index= ['Intercept', col])

            actual = lrp.get_standard_errors(reg, X, Y)

            np.testing.assert_almost_equal(actual.values, expected.values, decimal=places)
            self.assertListEqual(actual.index.tolist(), expected.index.tolist())

            places = 3

    def test_get_standard_errors_advertising_multi(self):
        '''
        Test the standard error using Advertising data, r summary, and ISLR pg. 74
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        reg = LinearRegression()
        reg.fit(X, Y)

        expected = pd.Series(data = np.array([0.311908, 0.001395, 0.008611, 0.005871]),
                             index = ['Intercept', 'TV', 'radio', 'newspaper'] )
        actual = lrp.get_standard_errors(reg, X, Y)

        np.testing.assert_almost_equal(actual.values, expected.values, decimal=6)
        self.assertListEqual(actual.index.tolist(), expected.index.tolist())


class TestGet_t_values(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.get_t_values().
    '''

    def test_get_t_values_advertising_simple(self):
        '''
        Test the standard error on simple linear regression using Advertising data and ISLR pg. 68, 72
        '''
        t_values = {'TV' : np.array([15.36, 17.67]),
                    'radio' : np.array([16.54, 9.92]),
                    'newspaper' : np.array([19.88, 3.30])}

        places = 2

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for col in t_values:
            X = pd.DataFrame(df[col])

            reg = LinearRegression()
            reg.fit(X, Y)

            expected = pd.Series(data=t_values[col],
                                 index=['Intercept', col])

            actual = lrp.get_t_values(reg, X, Y)

            np.testing.assert_almost_equal(actual.values, expected.values, decimal=places)
            self.assertListEqual(actual.index.tolist(), expected.index.tolist())

    def test_get_t_values_advertising_multi(self):
        '''
        Test the standard error using Advertising data, r summary, and ISLR pg. 74
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        reg = LinearRegression()
        reg.fit(X, Y)

        expected = pd.Series(data=np.array([9.422, 32.809, 21.893, -0.177]),
                             index=['Intercept', 'TV', 'radio', 'newspaper'])
        actual = lrp.get_t_values(reg, X, Y)

        np.testing.assert_almost_equal(actual.values, expected.values, decimal=3)
        self.assertListEqual(actual.index.tolist(), expected.index.tolist())


class TestGet_p_values(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.get_p_values().
    '''

    def test_get_p_values_advertising_simple(self):
        '''
        Test the standard error on simple linear regression using Advertising data and ISLR pg. 68, 72
        '''
        p_values = {'TV' : np.array([0.000000, 0.000000]),
                    'radio' : np.array([0.000000, 0.000000]),
                    'newspaper' : np.array([0.000000, 0.00115])}

        places = 5

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for col in p_values:
            X = pd.DataFrame(df[col])

            reg = LinearRegression()
            reg.fit(X, Y)

            expected = pd.Series(data=p_values[col],
                                 index=['Intercept', col])

            actual = lrp.get_p_values(reg, X, Y)

            np.testing.assert_almost_equal(actual.values, expected.values, decimal=places)
            self.assertListEqual(actual.index.tolist(), expected.index.tolist())

    def test_get_p_values_advertising_multi(self):
        '''
        Test the standard error using Advertising data, r summary, and ISLR pg. 74
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        reg = LinearRegression()
        reg.fit(X, Y)

        expected = pd.Series(data=np.array([0.0, 0.0, 0.0, 0.86]),
                             index=['Intercept', 'TV', 'radio', 'newspaper'])
        actual = lrp.get_p_values(reg, X, Y)

        np.testing.assert_almost_equal(actual.values, expected.values, decimal=2)
        self.assertListEqual(actual.index.tolist(), expected.index.tolist())


class TestSummary(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.summary().
    '''

    def test_summary_advertising_simple(self):
        '''
        Test summary on simple linear regression.
        '''
        columns = ['TV', 'radio', 'newspaper']

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for col in columns:
            X = pd.DataFrame(df[col])

            reg = LinearRegression()
            reg.fit(X, Y)

            summ_dfs = lrp.summary(reg, X, Y)

            actual = summ_dfs.shape
            expected = [2, 4]

            self.assertListEqual(list(actual), expected)

            # All p-values are essentially zero
            actual = summ_dfs['p-value'].sum()
            expected = 0.00

            self.assertAlmostEqual(actual, expected, places=2)

    def test_get_summary_advertising_multi(self):
        '''
        Test summary on multiple linear regression.
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        reg = LinearRegression()
        reg.fit(X, Y)

        summ_dfs = lrp.summary(reg, X, Y)

        actual = summ_dfs.shape
        expected = [4, 4]

        self.assertListEqual(list(actual), expected)

        # All p-values are essentially zero except newspaper
        actual = summ_dfs['p-value'].sum()
        expected = 0.90

        self.assertTrue(actual < expected)


if __name__ == '__main__':
    unittest.main()

