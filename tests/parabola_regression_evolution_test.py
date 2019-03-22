import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import parabola_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt


class TestParabolaRegression(unittest.TestCase):

    def test_best_creature_linear_regression_1_layer(self):
        evolution = evogression.evolution.CreatureEvolution('y', parabola_data, target_num_creatures=30000, num_cycles=30)
        best_creature, standardizer = evolution.return_best_creature()

        calculated_y_values = [standardizer.unstandardize_value('y', best_creature.calc_target(d)) for d in parabola_data]
        plt.scatter([d['x'] for d in parabola_data], [d['y'] for d in parabola_data])
        plt.scatter([d['x'] for d in parabola_data], calculated_y_values)
        plt.show()



if __name__ == '__main__':
    unittest.main()
