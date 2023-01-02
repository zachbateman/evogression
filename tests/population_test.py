import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import categorical_data, surface_3d_data
from pprint import pprint as pp
import matplotlib.pyplot as plt


class TestPopulationCateogry(unittest.TestCase):

    def test_population_category_2d(self):
        population = evogression.Population('y', categorical_data, split_parameter='cat', num_creatures=3000)
        y_test = [population.predict(d, 'pred')['pred'] for d in categorical_data]

        plt.scatter([pd['x'] for pd in categorical_data], [pd['y'] for pd in categorical_data], s=100)

        x_test_A = [i / 10 for i in range(0, 55)]
        y_test_A = [population.predict({'cat': 'A', 'x': x}, 'pred')['pred'] for x in x_test_A]

        x_test_B = [i / 10 for i in range(87, 135)]
        y_test_B = [population.predict({'cat': 'B', 'x': x}, 'pred')['pred'] for x in x_test_B]

        plt.scatter(x_test_A, y_test_A, s=10)
        plt.scatter(x_test_B, y_test_B, s=10)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.text(1, 17, 'Category A')
        plt.text(10.5, 17, 'Category B')
        plt.title('Category Regression - Population Test')

        plt.show()


    def test_population_continuous_3d(self):
        population = evogression.Population('z', surface_3d_data, num_creatures=5000, num_cycles=7, group_size=5, split_parameter='y', category_or_continuous='continuous')
        data = population.predict(surface_3d_data, 'z_predicted')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = [point_dict['x'] for point_dict in data]
        y = [point_dict['y'] for point_dict in data]
        z = [point_dict['z'] for point_dict in data]
        z_test = [point_dict['z_predicted'] for point_dict in data]

        ax.scatter3D(x, y, z)
        ax.scatter3D(x, y, z_test)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('Surface Regression - Continuous Population Test')

        plt.show()



if __name__ == '__main__':
    unittest.main()
