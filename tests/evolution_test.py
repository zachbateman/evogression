import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
from test_data_many_dimensions import data as many_d_data
from pprint import pprint as pp



class TestPredictionMethods(unittest.TestCase):
    def test_predict(self):
        '''
        Test ability of Evolution class to add target variable predictions
        to data samples.
        '''
        try:
            evolution = evogression.Evolution('y', linear_data, num_creatures=1000, num_cycles=5, use_multip=False)
            print(evolution.predict({'x': 5.2}))
            print(evolution.predict([{'x': -3.8}, {'x': 2.9}, {'x': 12.4}]))
            test_passed = True
        except TypeError:
            test_passed = False
        self.assertTrue(test_passed)

    def test_param_usage_counts(self):
        evolution = evogression.Evolution('Target', many_d_data, num_creatures=300, num_cycles=30, use_multip=False, optimize=False)
        pp(evolution.parameter_usefulness_count)
        d = dict(evolution.parameter_usefulness_count)
        self.assertTrue(len(d) > 1)
        self.assertTrue(sum(d.values()) > 3)

    def test_save_and_load(self):
        evolution = evogression.Evolution('y', linear_data, num_creatures=1000, num_cycles=5, use_multip=False)
        prediction1 = evolution.predict({'x': 5.2})
        evolution.save('linear_model')
        loaded_model = evogression.Evolution.load('linear_model')
        os.remove('linear_model.pkl')
        prediction2 = loaded_model.predict({'x': 5.2})
        self.assertTrue(prediction1 == prediction2)




if __name__ == '__main__':
    unittest.main(buffer=True)
