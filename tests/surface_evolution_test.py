import unittest
import sys
sys.path.insert(1, '..')
import evogression
from data_examples import surface_3d_data
import matplotlib.pyplot as plt


class Test3DSurfaceRegression(unittest.TestCase):
    def test_best_creature_3d(self):
        evolution = evogression.Evolution('z', surface_3d_data, creatures=30000, cycles=10, max_cpu=3)
        z_test = [evolution.predict(d, 'pred')['pred'] for d in surface_3d_data]
        x = [point_dict['x'] for point_dict in surface_3d_data]
        y = [point_dict['y'] for point_dict in surface_3d_data]
        z = [point_dict['z'] for point_dict in surface_3d_data]

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
