import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import parabola_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt


class TestLinearRegression(unittest.TestCase):

    def test_best_creature_linear_regression_1_layer(self):
        evolution = evogression.evolution.CreatureEvolution('y', parabola_data, initial_num_creatures=50000)



if __name__ == '__main__':
    unittest.main()
