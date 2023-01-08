import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
from test_data_many_dimensions import data as many_d_data


class TestPredictionMethods(unittest.TestCase):
    def test_predict(self):
        '''Test ability of Evolution class to add target variable predictions to data samples.'''
        evolution = evogression.Evolution('y', linear_data, num_creatures=1000, num_cycles=5)
        print(evolution.predict({'x': 5.2}))
        print(evolution.predict([{'x': -3.8}, {'x': 2.9}, {'x': 12.4}]))

    def test_param_usage_counts(self):
        evolution = evogression.Evolution('Target', many_d_data, num_creatures=300, num_cycles=30)
        print(evolution.parameter_usefulness_count)
        self.assertTrue(len(evolution.parameter_usefulness_count) > 1)
        self.assertTrue(sum(evolution.parameter_usefulness_count.values()) > 3)



if __name__ == '__main__':
    unittest.main(buffer=True)
