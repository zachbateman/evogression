import unittest
import sys
sys.path.insert(1, '..')
import evogression
from data_examples import surface_3d_data
import matplotlib.pyplot as plt


class Test3DSurfaceRegression(unittest.TestCase):
    def test_best_creature_3d(self):
        evolutions = evogression.random_population('z', surface_3d_data, num_creatures=25000, num_cycles=7, group_size=5)
        z_test = [d['pred'] for d in evolutions.predict(surface_3d_data, 'pred')]

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
