import unittest

import numpy as np
import pandas as pd

import linear_model_stats as lms

from sklearn.linear_model import LinearRegression

class TestRss(unittest.TestCase):
    '''
    This class tests linear_model_stats.rss().
    '''

    def test_rss_zero(self):
        '''
        .rss() should be zero due to being identical.
        '''
        Y1 = np.arange(100)
        Y2 = np.arange(100)

        actual = lms.rss(Y1, Y2)
        expected = 0

        self.assertEqual(actual, expected)

        Y1 = np.linspace(0, 100, 3)
        Y2 = np.linspace(0, 100, 3)

        self.assertEqual(actual, expected)


    def test_rss_basic(self):
        '''
        This is just a basic test with a small amount of variables that can be easily calculated.
        '''
        Y1 = np.array([1, 3, 7])
        Y2 = np.array([0, 6, 9])

        actual = lms.rss(Y1, Y2)
        expected = 14

        self.assertEqual(actual, expected)

    def test_rss_len(self):
        '''
        This test calculates RSS in a way where RSS can be calculated as a a factor of the length.
        '''
        Y1 = np.arange(100)

        for i in range(1, 15):
            Y2 = Y1 + i
            fac = i**2

            actual = lms.rss(Y1, Y2)
            expected = Y1.size * fac

            self.assertEqual(actual, expected)

class TestRse(unittest.TestCase):
    '''
    This class tests linear_model_stats.rse().
    '''

    def test_rse_basic(self):
        '''
        This test is testing rse in a basic case can be easily calculated manually.
        '''
        Y1 = np.array([1, 3, 7, 11])
        Y2 = np.array([0, 6, 9, 11])

        # rss = 14
        rss = lms.rss(Y1, Y2)

        actual = lms.rse(rss, Y1.size, 2)

        # Should be sqrt of 14
        expected = 3.74165738677

        self.assertAlmostEqual(actual, expected, places=7)

        actual = lms.rse(rss, Y1.size, 1)

        # Should be sqrt of 7
        expected = 2.64575131106

        self.assertAlmostEqual(actual, expected, places=7)

    def test_rse_seven_factorial(self):
        '''
        A slightly more complicated basic example where the rss is seven factorial.
        '''
        rss = 5040
        n = 9

        for p in range(1, 8):
            actual = lms.rse(rss, n, p)
            x = rss / (n - p - 1)
            expected = np.sqrt(x)

            self.assertAlmostEqual(actual, expected, places=7)

class TestRse_arr(unittest.TestCase):
    '''
    This class tests linear_model_stats.rse_arr().
    '''

    def test_rse_arr_basic(self):
        '''
        This test is testing rse in a basic case that can be easily calculated manually.
        '''
        Y1 = np.array([1, 3, 7, 11])
        Y2 = np.array([0, 6, 9, 11])

        actual = lms.rse_arr(Y1, Y2, 2)

        # Should be sqrt of 14
        expected = 3.74165738677

        self.assertAlmostEqual(actual, expected, places=7)

        actual = lms.rse_arr(Y1, Y2, 1)

        # Should be sqrt of 7
        expected = 2.64575131106

        self.assertAlmostEqual(actual, expected, places=7)

class TestRse_arr(unittest.TestCase):
    '''
    This class tests linear_model_stats.rse_arr().
    '''

    def test_rse_arr_basic(self):
        '''
        This test is testing rse in a basic case that can be easily calculated manually.
        '''
        Y1 = np.array([1, 3, 7, 11])
        Y2 = np.array([0, 6, 9, 11])

        actual = lms.rse_arr(Y1, Y2, 2)

        # Should be sqrt of 14
        expected = 3.74165738677

        self.assertAlmostEqual(actual, expected, places=7)

        actual = lms.rse_arr(Y1, Y2, 1)

        # Should be sqrt of 7
        expected = 2.64575131106

        self.assertAlmostEqual(actual, expected, places=7)

class TestStandard_error(unittest.TestCase):
    '''
    This class tests linear_model_stats.standard_error().
    '''

    def test_standard_error_basic(self):
        '''
        This test is testing standard error in a basic case that can be easily calculated manually.
        '''

        var = 13
        X = np.arange(0, 11, 2)

        actual = lms.standard_error(var, X)
        expected = 0.43094580368

        self.assertAlmostEqual(actual, expected, places=7)

    def test_standard_error_advertising(self):
        '''
        This is testing standard error using data from ISLR pg. 68 and pg. 72 to check against.
        '''
        standard_error_data = [('TV', 0.0027), ('radio', 0.020), ('newspaper', 0.017)]
        places = 4

        df = pd.read_csv('Advertising.csv')

        # Set Y to sales
        Y = df['sales']

        # Iterate over standard error data from ISLR and test each
        for i in range(len(standard_error_data )):
            stan_e_tup = standard_error_data[i]

            # Set X to column
            X = pd.DataFrame(df[stan_e_tup[0]])

            # Create a linear model and train
            reg = LinearRegression()
            reg.fit(X, Y)

            # Get predictions for Y and then calculate rse from this.
            Y_pred = reg.predict(X)
            rss = lms.rss(Y.values, Y_pred)
            rse = lms.rse(rss, Y.values.size, X.columns.size)
            var = rse**2

            actual = lms.standard_error(var, X[stan_e_tup[0]].values)
            expected = stan_e_tup[1]

            self.assertAlmostEqual(actual, expected, places=places)
            places = 3

    def test_standard_error_advertising_full_X(self):
        '''
        This is testing standard error for all three columns using data from ISLR pg. 74 to check against
        '''
        standard_error_data = [('TV', 0.001395), ('radio', 0.008611), ('newspaper', 0.005871)]
        places = 4

        df = pd.read_csv('Advertising.csv')

        # Set Y to sales
        Y = df['sales']

        # Set X to newspaper
        X = pd.DataFrame(df[['TV', 'radio', 'newspaper']])

        # Create a linear model and train
        reg = LinearRegression()
        reg.fit(X, Y)

        # Get predictions for Y
        Y_pred = reg.predict(X)
        rss = lms.rss(Y.values, Y_pred)
        rse = lms.rse(rss, Y.values.size, X.columns.size)
        var = rse**2

        for i in range(len(standard_error_data)):
            stan_e_tup = standard_error_data[i]

            standard_error = lms.standard_error(var, X[stan_e_tup[0]].values)

            actual = standard_error
            expected = stan_e_tup[1]

            print(stan_e_tup[0])
            print('Actual:\t\t', actual)
            print('Expected:\t', expected)
            print()

            # self.assertAlmostEqual(actual, expected, places=4)

class TestT_stat(unittest.TestCase):
    '''
    This class tests linear_model_stats.t_stat().
    '''

    def test_t_stat_basic(self):
        '''
        This test is testing t statistic in a basic case that can be easily calculated manually.
        '''

        actual = lms.t_stat(25, 100)
        expected = 0.25

        self.assertEqual(actual, expected)

    def test_t_stat_advertising(self):
        '''
        This is testing standard error using data from ISLR pg. 68 and pg. 72 to check against.
        '''
        t_stat_data = [('TV', 17.67), ('radio', 9.92), ('newspaper', 3.30)]
        places = 2

        df = pd.read_csv('Advertising.csv')

        # Set Y to sales
        Y = df['sales']

        # Iterate over standard error data from ISLR and test each
        for i in range(len(t_stat_data )):
            t_stat_tup = t_stat_data[i]

            # Set X to column
            X = pd.DataFrame(df[t_stat_tup[0]])

            # Create a linear model and train
            reg = LinearRegression()
            reg.fit(X, Y)

            # Get predictions for Y and then calculate rse from this.
            Y_pred = reg.predict(X)
            rss = lms.rss(Y.values, Y_pred)
            rse = lms.rse(rss, Y.values.size, X.columns.size)
            var = rse**2
            standard_error = lms.standard_error(var, X[t_stat_tup[0]].values)

            actual = lms.t_stat(reg.coef_[0], standard_error)
            expected = t_stat_tup[1]

            self.assertAlmostEqual(actual, expected, places=places)

class TestTwo_sided_p_value(unittest.TestCase):
    '''
    This class tests linear_model_stats.p_value().
    '''

    def test_p_value_advertising_newspaper(self):
        '''
        This is testing p-value for newspaper using data from ISLR pg. 72 to check against
        '''
        places = 5

        df = pd.read_csv('Advertising.csv')

        # Set Y to sales
        Y = df['sales']

        # Set X to newspaper
        X = pd.DataFrame(df['newspaper'])

        # Create a linear model and train
        reg = LinearRegression()
        reg.fit(X, Y)

        # Get predictions for Y and then calculate rse from this.
        Y_pred = reg.predict(X)
        rss = lms.rss(Y.values, Y_pred)
        rse = lms.rse(rss, Y.values.size, X.columns.size)
        var = rse**2
        standard_error = lms.standard_error(var, X['newspaper'].values)
        t_stat = lms.t_stat(reg.coef_[0], standard_error)

        actual = lms.two_sided_p_value(t_stat, Y.size)
        expected = 0.00115

        self.assertAlmostEqual(actual, expected, places=places)

    def test_p_value_advertising_full_X(self):
        '''
        This is testing p-value for newspaper using data from ISLR pg. 74 to check against
        '''
        places = 5

        df = pd.read_csv('Advertising.csv')

        # Set Y to sales
        Y = df['sales']

        # Set X to newspaper
        X = pd.DataFrame(df[['TV', 'radio', 'newspaper']])

        # Create a linear model and train
        reg = LinearRegression()
        reg.fit(X, Y)

        # Get predictions for Y and then calculate rse from this.
        Y_pred = reg.predict(X)
        rss = lms.rss(Y.values, Y_pred)
        rse = lms.rse(rss, Y.values.size, X.columns.size)
        var = rse**2
        standard_error = lms.standard_error(var, X['newspaper'].values)
        t_stat = lms.t_stat(reg.coef_[2], standard_error)

        actual = standard_error
        expected = -0.0059

        self.assertAlmostEqual(actual, expected, places=4)

        # actual = lms.two_sided_p_value(t_stat, Y.size)
        # expected = 0.8599
        #
        # self.assertAlmostEqual(actual, expected, places=places)

class TestTss(unittest.TestCase):
    '''
    This class tests linear_model_stats.tss().
    '''

    def test_tss_zero(self):
        '''
        .tss() should be zero due to having all the same number.
        '''
        expected = 0

        for i in range(20):
            Y1 = np.array([i] * 50)

            actual = lms.tss(Y1)
            self.assertEqual(actual, expected)

    def test_tss_basic(self):
        '''
        This is just a basic test of tss with a small amount of variables that can be easily calculated.
        '''
        Y1 = np.array([5, 3, 7])
        actual = lms.tss(Y1)
        expected = 8
        self.assertEqual(actual, expected)

        Y2 = np.array([5, 12, 43])
        actual = lms.tss(Y2)
        expected = 818
        self.assertEqual(actual, expected)

        Y3 = np.array([2, 3, 5, 7])
        actual = lms.tss(Y3)
        expected = 14.75
        self.assertEqual(actual, expected)

class TestF_stat(unittest.TestCase):
    '''
    This class tests linear_model_stats.f_stat().
    '''

    def test_f_stat_basic(self):
        '''
        This is testing f-stat for using data that can be tested automatically.
        '''
        tss = 100
        rss = 25
        n = 50
        p = 5
        f_stat = lms.f_stat(tss, rss, n, p)

        actual = f_stat
        expected = 26.4

        self.assertEqual(actual, expected)

    def test_f_stat_advertising(self):
        '''
        This is testing f-stat for using data from ISLR pg. 76 to check against.
        '''
        df = pd.read_csv('Advertising.csv')

        # Set Y to sales
        Y = df['sales']

        # Set X to newspaper
        X = pd.DataFrame(df[['TV', 'radio', 'newspaper']])

        # Create a linear model and train
        reg = LinearRegression()
        reg.fit(X, Y)

        # Get predictions for Y and then calculate items for t-stat
        Y_pred = reg.predict(X)
        tss = lms.tss(Y)
        rss = lms.rss(Y.values, Y_pred)
        f_stat = lms.f_stat(tss, rss, Y.size, X.columns.size)

        actual = round(f_stat)
        expected = 570

        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()

