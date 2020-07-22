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
        evolutions = evogression.random_population(surface_3d_data, 'z', num_creatures=5000, num_cycles=5, group_size=15, optimize=5, use_multip=False)
        z_test = [sum(e.predict(d, 'pred')['pred'] for e in evolutions) / len(evolutions) for d in surface_3d_data]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = [point_dict['x'] for point_dict in surface_3d_data]
        y = [point_dict['y'] for point_dict in surface_3d_data]
        z = [point_dict['z'] for point_dict in surface_3d_data]

        ax.scatter3D(x, y, z)
        ax.scatter3D(x, y, z_test)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('Surface Regression - Random Population Test')

        plt.show()



if __name__ == '__main__':
    unittest.main()
