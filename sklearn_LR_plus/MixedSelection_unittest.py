import unittest

import numpy as np
import pandas as pd
import itertools

import sklearn_LR_plus as lrp

from sklearn.linear_model import LinearRegression


class TestSimple_linear_regressions(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.MixedSelection.simple_linear_regressions().
    '''

    def test_simple_linear_regressions_advertising_simple(self):
        '''
        Test simple_linear_regressions() using Advertising data and ISLR pg. 68, 72
        '''
        standard_errors = {'TV' : np.array([0.4578, 0.0027]),
                           'radio' : np.array([0.563, 0.020]),
                           'newspaper' : np.array([0.621, 0.017])}

        t_values = {'TV' : np.array([15.36, 17.67]),
                    'radio' : np.array([16.54, 9.92]),
                    'newspaper' : np.array([19.88, 3.30])}

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        # Why does this not work!
        ms = lrp.MixedSelection(X, Y)
        simp_LRs = ms.simple_linear_regressions()

        for lr in simp_LRs:
            col = lr.features[0]

            expected = pd.Series(data=standard_errors[col],
                                 index=['Intercept', col])
            actual = lr.get_standard_errors()

            pd.testing.assert_series_equal(actual.round(3), expected.round(3))

            expected = pd.Series(data=t_values[col],
                                 index=['Intercept', col])
            actual = lr.get_t_values()

            pd.testing.assert_series_equal(actual.round(2), expected.round(2))

        for i in range(1, len(simp_LRs)):
            self.assertTrue(simp_LRs[i - 1].get_rss() <= simp_LRs[i].get_rss())


class TestMixed_selection(unittest.TestCase):
    '''
    This class tests sklearn_LR_plus.MixedSelection.mixed_selection().
    '''
    def test_mixed_selection_advertising_simple(self):
        '''
        Test  on simple linear regression using Advertising data and ISLR pg. 68, 72
        '''
        df = pd.read_csv('Advertising.csv')

        Y = df['sales']
        X = df[['TV', 'radio', 'newspaper']]

        # Why does this not work!
        ms = lrp.MixedSelection(X, Y)
        lrm = ms.mixed_selection()

        expected = ['TV', 'radio']
        actual = lrm.features
        actual.sort()

        self.assertListEqual(expected, actual)


    def test_mixed_selection_advertising_permutations(self):
        '''
        Test mixed selection using Advertising data for all permutations on the order of columns and ISLR pg. 68, 72
        Thinking about this, this should be overkill becuase the simple linear regressions should be sorted by rss so
        it ends up being much less of a test of .mixed_selection().
        '''

        df = pd.read_csv('Advertising.csv')

        Y = df['sales']

        for cols in itertools.permutations(['TV', 'radio', 'newspaper']):
            X = df[list(cols)]

            # Why does this not work!
            ms = lrp.MixedSelection(X, Y)
            lrm = ms.mixed_selection()

            expected = ['TV', 'radio']
            actual = lrm.features
            actual.sort()

            self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()