import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data


class TestNoneFilling(unittest.TestCase):

    def test_none_fill(self):
        '''
        Test ability of CreatureEvolution class to handle None values in input data
        by filling them with the median of populated values.
        '''
        test_data = linear_data
        test_data[1] = {'x': 5, 'y': None}
        test_data[4] = {'x': None, 'y': 11.2}
        test_data[5] = {'x': float('nan'), 'y': 11.2}
        test_data[6] = {'x': None, 'y': float('nan')}
        try:
            evolution = evogression.Evolution('y', test_data, num_creatures=100, num_cycles=1, use_multip=False)
            test_passed = True
        except TypeError:
            test_passed = False
        self.assertTrue(test_passed)



if __name__ == '__main__':
    unittest.main(buffer=True)
