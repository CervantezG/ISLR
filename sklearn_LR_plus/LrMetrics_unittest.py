import unittest

import numpy as np
import pandas as pd

import sklearn_LR_plus as lrp

from sklearn.linear_model import LinearRegression

class TestContructor(unittest.TestCase):
    '''
    This class tests the sklearn_LR_plus.LrMetrics constructor.
    '''

    def testConstructor(self):
        '''
        Test that results are the same if initializing using a constructor without reg and with reg.
        '''
        standard_errors = ['TV', 'radio', 'newspaper']

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for col in standard_errors:
            X = pd.DataFrame(df[col])

            reg = LinearRegression()
            reg.fit(X, Y)
            lrm1 = lrp.LrMetrics(X, Y, reg)

            lrm2 = lrp.LrMetrics(X, Y)

            self.assertEqual(lrm1.reg.score(X, Y), lrm2.reg.score(X, Y))
            self.assertEqual(lrm1.reg.coef_, lrm2.reg.coef_)
            self.assertEqual(lrm1.reg.intercept_, lrm2.reg.intercept_)


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

            expected = pd.Series(data=standard_errors[col],
                                 index=['Intercept', col])

            lrm = lrp.LrMetrics(X, Y)

            actual = lrm.get_standard_errors()

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

        expected = pd.Series(data=np.array([0.311908, 0.001395, 0.008611, 0.005871]),
                             index=['Intercept', 'TV', 'radio', 'newspaper'])

        lrm = lrp.LrMetrics(X, Y)

        actual = lrm.get_standard_errors()

        np.testing.assert_almost_equal(actual.values, expected.values, decimal=6)
        self.assertListEqual(actual.index.tolist(), expected.index.tolist())


class TestGet_t_values(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.LrMetrics.get_t_values().
    '''

    def test_get_t_values_advertising_simple(self):
        '''
        Test the t values on simple linear regression using Advertising data and ISLR pg. 68, 72
        '''
        t_values = {'TV' : np.array([15.36, 17.67]),
                    'radio' : np.array([16.54, 9.92]),
                    'newspaper' : np.array([19.88, 3.30])}

        places = 2

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for col in t_values:
            X = pd.DataFrame(df[col])

            expected = pd.Series(data=t_values[col],
                                 index=['Intercept', col])

            lrm = lrp.LrMetrics(X, Y)

            actual = lrm.get_t_values()

            np.testing.assert_almost_equal(actual.values, expected.values, decimal=places)
            self.assertListEqual(actual.index.tolist(), expected.index.tolist())

    def test_get_t_values_advertising_multi(self):
        '''
        Test the t values using Advertising data, r summary, and ISLR pg. 74
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        expected = pd.Series(data=np.array([9.422, 32.809, 21.893, -0.177]),
                             index=['Intercept', 'TV', 'radio', 'newspaper'])

        lrm = lrp.LrMetrics(X, Y)

        actual = lrm.get_t_values()

        np.testing.assert_almost_equal(actual.values, expected.values, decimal=3)
        self.assertListEqual(actual.index.tolist(), expected.index.tolist())

#
class TestGet_p_values(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.LrMetrics.get_p_values().
    '''

    def test_get_p_values_advertising_simple(self):
        '''
        Test the p values on simple linear regression using Advertising data and ISLR pg. 68, 72
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

            lrm = lrp.LrMetrics(X, Y)
            actual = lrm.get_p_values()

            np.testing.assert_almost_equal(actual.values, expected.values, decimal=places)
            self.assertListEqual(actual.index.tolist(), expected.index.tolist())

    def test_get_p_values_advertising_multi(self):
        '''
        Test the p values using Advertising data, r summary, and ISLR pg. 74
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        expected = pd.Series(data=np.array([0.0, 0.0, 0.0, 0.86]),
                             index=['Intercept', 'TV', 'radio', 'newspaper'])
        lrm = lrp.LrMetrics(X, Y)
        actual = lrm.get_p_values()

        np.testing.assert_almost_equal(actual.values, expected.values, decimal=2)
        self.assertListEqual(actual.index.tolist(), expected.index.tolist())


class TestGet_coef_tests(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.LrMetrics.get_coef_tests().
    '''
    def test_get_coef_tests_advertising_simple(self):
        '''
        Test .get_coef_tests() on simple linear regression.
        '''
        columns = ['TV', 'radio', 'newspaper']

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for col in columns:
            X = pd.DataFrame(df[col])

            lrm = lrp.LrMetrics(X, Y)
            coef_df = lrm.get_coef_tests()

            actual = coef_df.shape
            expected = [2, 4]

            self.assertListEqual(list(actual), expected)

            # All p-values are essentially zero
            actual = coef_df['p-value'].sum()
            expected = 0.00

            self.assertAlmostEqual(actual, expected, places=2)

    def test_get_coef_tests_advertising_multi(self):
        '''
        Test .get_coef_tests() on multiple linear regression.
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        lrm = lrp.LrMetrics(X, Y)

        # Test coef_tests
        coef_df = lrm.get_coef_tests()

        actual = coef_df.shape
        expected = [4, 4]

        self.assertListEqual(list(actual), expected)

        # All p-values are essentially zero except newspaper which is ~ 0.85
        actual = coef_df['p-value'].sum()
        expected_high = 0.90
        expected_low = 0.80
        self.assertTrue(actual < expected_high)
        self.assertTrue(actual > expected_low)


class TestGet_residuals_data(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.LrMetrics.get_residuals_data().
    '''
    def test_get_residuals_data_advertising_multi(self):
        '''
        Test .get_residuals_data() on multiple linear regression.
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        lrm = lrp.LrMetrics(X, Y)

        # Test residuals
        expected = pd.Series(data=[-8.8277, -0.8908, 0.2418, 1.1893, 2.8292],
                             index=['min', '25%', '50%', '75%', 'max'])
        expected.name = 'Residuals'

        lrm = lrp.LrMetrics(X, Y)
        actual = lrm.get_residuals_data().drop('std')

        pd.testing.assert_series_equal(actual, expected, check_less_precise=4)

        # # Test general
        # expected = pd.Series(data=[1.686, 570.3, -99, 0.8972, -99],
        #                      index=['rse', 'f-stat', 'p-value', 'R^2', 'adj_R^2'])
        # expected.name = 'General'
        #
        # actual = summary['general']
        #
        # self.assertAlmostEqual(actual['f-stat'], expected['f-stat'], places=1)
        #
        # expected.drop('f-stat', inplace=True)
        # actual.drop('f-stat', inplace=True)
        #
        # pd.testing.assert_series_equal(actual, expected, check_less_precise=3)

class TestGet_general_data(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.LrMetrics.get_general_data().
    '''
    def test_get_general_data_advertising_multi(self):
        '''
        Test .get_general_data() on multiple linear regression.
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        lrm = lrp.LrMetrics(X, Y)

        # Test general
        expected = pd.Series(data=[1.686, 570.3, -99, 0.8972, -99],
                             index=['rse', 'f-stat', 'p-value', 'R^2', 'adj_R^2'])
        expected.name = 'General'

        actual = lrm.get_general_data()

        self.assertAlmostEqual(actual['f-stat'], expected['f-stat'], places=1)

        expected.drop('f-stat', inplace=True)
        actual.drop('f-stat', inplace=True)

        pd.testing.assert_series_equal(actual, expected, check_less_precise=3)


class TestGet_high_p_features(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.LrMetrics.get_general_data().
    '''
    def test_get_high_p_features_advertising_multi(self):
        '''
        Test the p values using Advertising data, r summary, and ISLR pg. 74
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        # Test that values are known
        expected = pd.Series(data=np.array([0.0, 0.0, 0.0, 0.86]),
                             index=['Intercept', 'TV', 'radio', 'newspaper'])
        lrm = lrp.LrMetrics(X, Y)
        actual = lrm.get_p_values()

        np.testing.assert_almost_equal(actual.values, expected.values, decimal=2)
        self.assertListEqual(actual.index.tolist(), expected.index.tolist())

        # Test with default q
        expected = ['newspaper']
        actual = lrm.get_high_p_features()

        self.assertListEqual(expected, actual)

        # Test with q=0.5
        expected = ['newspaper']
        actual = lrm.get_high_p_features(0.5)

        self.assertListEqual(expected, actual)

        # Test with q=0.9
        expected = []
        actual = lrm.get_high_p_features(0.9)

        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

