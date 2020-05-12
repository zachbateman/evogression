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
        evolution = evogression.Evolution('z', df, num_creatures=500, num_cycles=5, use_multip=False, optimize=3)
        predicted = evolution.predict(df)
        self.assertTrue(type(predicted) == pandas.DataFrame)
        self.assertTrue(len(predicted.columns) == 4)
        self.assertTrue('z_PREDICTED' in predicted.columns)

        single_prediction = evolution.predict({'x': 0, 'y': 0}, 'z_test')
        self.assertTrue('z_test' in single_prediction)
        self.assertTrue(len(single_prediction) == 3)





if __name__ == '__main__':
    unittest.main(buffer=True)
