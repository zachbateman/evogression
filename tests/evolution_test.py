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

    def test_max_layers(self):
        evolution = evogression.Evolution('y', linear_data, max_layers=2, num_creatures=10000, num_cycles=1, use_multip=False)
        single_layer, double_layer = 0, 0
        for cr in evolution.creatures:
            if cr.layers == 1:
                single_layer += 1
            elif cr.layers == 2:
                double_layer += 1
            else:
                print(cr.layers)
        print(f'{single_layer}')
        print(f'{double_layer}')
        print(single_layer+double_layer)
        self.assertTrue(max((cr.layers for cr in evolution.creatures)) <= 2)

    def test_param_usage_counts(self):
        evolution = evogression.Evolution('Target', many_d_data, num_creatures=300, num_cycles=30, use_multip=False, optimize=False)
        pp(evolution.parameter_usefulness_count)
        d = dict(evolution.parameter_usefulness_count)
        self.assertTrue(len(d) > 1)
        self.assertTrue(sum(d.values()) > 3)
        self.assertTrue(True)

    def test_bad_data_N(self):
        self.assertRaises(evogression.evolution.InputDataFormatError,
            evogression.Evolution, 'a', [{'a': 5, 'b': 5, 'N': 6}, {'a': 6, 'b': 6, 'N': 7}])

    def test_bad_data_T(self):
        self.assertRaises(evogression.evolution.InputDataFormatError,
            evogression.Evolution, 'a', [{'a': 5, 'b': 5, 'T': 6}, {'a': 6, 'b': 6, 'T': 7}])




if __name__ == '__main__':
    unittest.main(buffer=True)
