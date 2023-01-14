import unittest
import sys
sys.path.insert(1, '..')
import evogression
from data_examples import linear_data, surface_3d_data, many_dimension_data
import pandas


class TestPredictionMethods(unittest.TestCase):
    def test_predict(self):
        '''Test ability of Evolution class to add target variable predictions to data samples.'''
        evolution = evogression.Evolution('y', linear_data, creatures=1000, cycles=5)
        print(evolution.predict({'x': 5.2}))
        print(evolution.predict([{'x': -3.8}, {'x': 2.9}, {'x': 12.4}]))

    def test_param_usage_counts(self):
        evolution = evogression.Evolution('Target', many_dimension_data, creatures=300, cycles=30)
        print(evolution.parameter_usefulness_count)
        self.assertTrue(len(evolution.parameter_usefulness_count) > 1)
        self.assertTrue(sum(evolution.parameter_usefulness_count.values()) > 3)


class TestData(unittest.TestCase):
    def test_load_and_predict(self):
        '''Test ability of Evolution class to handle pandas DataFrames'''
        df = pandas.DataFrame(surface_3d_data)
        evolution = evogression.Evolution('z', df, creatures=500, cycles=5)
        predicted = evolution.predict(df)
        self.assertTrue(isinstance(predicted, pandas.DataFrame))
        self.assertTrue(len(predicted.columns) == 4)
        self.assertTrue('z_PREDICTED' in predicted.columns)

        single_prediction = evolution.predict({'x': 0, 'y': 0}, 'z_test')
        self.assertTrue('z_test' in single_prediction)
        self.assertTrue(len(single_prediction) == 3)

    def test_dirty_data(self):
        data1 = [
            {'x': 6, 'y': '7.1', 'z': 0.56},
        ]
        self.assertRaises(evogression.InputDataFormatError, lambda: evogression.Evolution('z', data1))

        data2 = [
            {'x': 5, 'y': 8.1, 'z': 0.82},
            {'x': None, 'y': 5.1, 'z': 0.75},
            {'x': 9, 'y': float('nan'), 'z': 0.87},
            {'x': 11, 'y': 3.1, 'z': 0.91},
        ]
        model = evogression.Evolution('z', data2, creatures=500, cycles=5)
        predicted = model.predict(data2)
        self.assertTrue('z_PREDICTED' in predicted[0])

    def test_none_fill(self):
        '''
        Test ability to handle None values in input data
        by filling them with the median of populated values.
        '''
        test_data = linear_data
        test_data[1] = {'x': 5, 'y': None}
        test_data[4] = {'x': None, 'y': 11.2}
        test_data[5] = {'x': float('nan'), 'y': 11.2}
        test_data[6] = {'x': None, 'y': float('nan')}
        evogression.Evolution('y', test_data)



if __name__ == '__main__':
    unittest.main(buffer=True)
