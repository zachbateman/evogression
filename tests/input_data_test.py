import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import surface_3d_data
import pandas


class TestDataFrame(unittest.TestCase):
    def test_load_and_predict(self):
        '''
        Test ability of Evolution class to handle pandas DataFrames.
        '''
        df = pandas.DataFrame(surface_3d_data)
        evolution = evogression.Evolution('z', df, num_creatures=500, num_cycles=5)
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
        model = evogression.Evolution('z', data2, num_creatures=500, num_cycles=5)
        predicted = model.predict(data2)
        self.assertTrue('z_PREDICTED' in predicted[0])



if __name__ == '__main__':
    unittest.main(buffer=True)
