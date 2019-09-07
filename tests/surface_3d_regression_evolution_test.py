import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import surface_3d_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import random
random.seed(10)  # for reproducing the same plot


class Test3DSurfaceRegression(unittest.TestCase):

    def test_best_creature_3d(self):
        evolution = evogression.evolution.CreatureEvolutionFittest('z', surface_3d_data, target_num_creatures=30000, num_cycles=10)

        x = [point_dict['x'] for point_dict in surface_3d_data]
        y = [point_dict['y'] for point_dict in surface_3d_data]
        z = [point_dict['z'] for point_dict in surface_3d_data]

        standardized_3d_data = [evolution.standardizer.convert_parameter_dict_to_standardized(d) for d in surface_3d_data]
        z_test = [evolution.standardizer.unstandardize_value('z', evolution.best_creature.calc_target(d)) for d in standardized_3d_data]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter3D(x, y, z)
        ax.scatter3D(x, y, z_test)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('Surface Regression - Evolution Test')

        plt.show()



if __name__ == '__main__':
    unittest.main()
