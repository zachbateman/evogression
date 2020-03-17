import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data


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
        print(f'{single_layer=}')
        print(f'{double_layer=}')
        print(single_layer+double_layer)

        self.assertTrue(max((cr.layers for cr in evolution.creatures)) <= 2)



if __name__ == '__main__':
    unittest.main()
