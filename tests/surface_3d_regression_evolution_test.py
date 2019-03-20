import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import surface_3d_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt


class Test3DSurfaceRegression(unittest.TestCase):

    def test_best_creature_3d(self):
        evolution = evogression.evolution.CreatureEvolution('z', surface_3d_data, target_num_creatures=500)



if __name__ == '__main__':
    unittest.main()
