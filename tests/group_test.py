import unittest
import sys
sys.path.insert(1, '..')
import evogression
from data_examples import linear_data, categorical_data, surface_3d_data, many_dimension_data
import matplotlib.pyplot as plt


class TestGroup(unittest.TestCase):
    def test_parameter_usage_count(self):
        group = evogression.evolution_group('Target', many_dimension_data, creatures=500, cycles=5, group_size=10)
        for model in group.models:
            print(model.parameter_usefulness_count)
            self.assertTrue(len(model.parameter_usefulness_count) > 0)
        print(group.parameter_usage)
        self.assertTrue(len(group.parameter_usage) > 0)

    def test_evolution_group(self):
        group = evogression.evolution_group('y', linear_data, creatures=3000, group_size=30, cycles=10)
        calculation_x_values = [i / 2 for i in range(6, 27)]
        for evo in group.models:
            calculated_y_values = [evo.predict({'x': x}, 'pred')['pred'] for x in calculation_x_values]
            plt.plot(calculation_x_values, calculated_y_values, alpha=0.1)
        calculated_y_values = [group.predict({'x': x}, 'pred')['pred'] for x in calculation_x_values]

        plt.plot(calculation_x_values, calculated_y_values,  'g--', alpha=1.0)
        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression - Group Composite Prediction Test')
        plt.show()


class TestPopulationCateogry(unittest.TestCase):
    def test_population_category_2d(self):
        population = evogression.Population('y', categorical_data, split_parameter='cat', creatures=3000)
        y_test = [population.predict(d, 'pred')['pred'] for d in categorical_data]
        x_test_A = [i / 10 for i in range(0, 55)]
        y_test_A = [population.predict({'cat': 'A', 'x': x}, 'pred')['pred'] for x in x_test_A]
        x_test_B = [i / 10 for i in range(87, 135)]
        y_test_B = [population.predict({'cat': 'B', 'x': x}, 'pred')['pred'] for x in x_test_B]

        plt.scatter([pd['x'] for pd in categorical_data], [pd['y'] for pd in categorical_data], s=100)
        plt.scatter(x_test_A, y_test_A, s=10)
        plt.scatter(x_test_B, y_test_B, s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.text(1, 17, 'Category A')
        plt.text(10.5, 17, 'Category B')
        plt.title('Category Regression - Population Test')
        plt.show()

    def test_population_continuous_3d(self):
        population = evogression.Population('z', surface_3d_data, creatures=3000, cycles=5, group_size=5, split_parameter='y', category_or_continuous='continuous')
        data = population.predict(surface_3d_data, 'z_predicted')
        x = [point_dict['x'] for point_dict in data]
        y = [point_dict['y'] for point_dict in data]
        z = [point_dict['z'] for point_dict in data]
        z_test = [point_dict['z_predicted'] for point_dict in data]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x, y, z)
        ax.scatter3D(x, y, z_test)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('Surface Regression - Continuous Population Test')
        plt.show()


if __name__ == '__main__':
    unittest.main(buffer=True)
