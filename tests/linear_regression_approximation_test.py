import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
import matplotlib
import matplotlib.pyplot as plt


class TestLinearRegression(unittest.TestCase):

    @classmethod
    def setUpClass(TestLinearRegression):
        TestLinearRegression.creatures =[evogression.EvogressionCreature('y', layers=1, full_parameter_example={'x': None, 'y': None}) for _ in range(50000)]

    def test_best_creature_linear_regression_1_layer(self):
        best_error = 10000
        best_creature = None
        for cr_index, creature in enumerate(self.creatures):
            error = 0
            for d in linear_data:
                target_calc = creature.calc_target({'x': d['x']})
                error += abs(target_calc - d['y']) ** 2
            if error < best_error:
                best_error = error
                best_creature = creature
                print(f'New best creature found!  Index: {cr_index}')
        # now have "best_creature"
        calculation_x_values = [i / 2 for i in range(6, 25)]
        calculated_y_values = [best_creature.calc_target({'x': x}) for x in calculation_x_values]

        print('\nBest creature found!')
        print(f'  linear regression error^2: 50.8')
        print(f'  creature total error^2:    {round(best_error, 1)}')

        print('\nModifiers:')
        print(best_creature)
        print('\n'*2)

        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.plot(calculation_x_values, calculated_y_values, 'g--')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression - Brute Force Single Layer Test')
        plt.show()


class TestLinearRegressionLayers(unittest.TestCase):

    @classmethod
    def setUpClass(TestLinearRegressionLayers):
        TestLinearRegressionLayers.creatures = [evogression.EvogressionCreature('y', full_parameter_example={'x': None, 'y': None}, layers=3) for _ in range(100000)]

    def test_best_creature_linear_regression_layers(self):
        best_error = 10000
        best_creature = None
        for cr_index, creature in enumerate(self.creatures):
            error = 0
            for d in linear_data:
                target_calc = creature.calc_target({'x': d['x']})
                error += abs(target_calc - d['y']) ** 2
            if error < best_error:
                best_error = error
                best_creature = creature
                print(f'New best creature found!  Index: {cr_index}')
        # now have "best_creature"
        calculation_x_values = [i / 2 for i in range(6, 25)]
        calculated_y_values = [best_creature.calc_target({'x': x}) for x in calculation_x_values]

        print('\nBest creature found!')
        print(f'  linear regression error^2: 50.8')
        print(f'  creature total error^2:    {round(best_error, 1)}')

        print('\nModifiers:')
        print(best_creature)
        print('\n'*2)

        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.plot(calculation_x_values, calculated_y_values, 'g--')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression - Brute Force Multiple Layers Test')
        plt.show()


class TestLinearRegressionEvolution(unittest.TestCase):
    def test_best_creature_evolution(self):
        evolution = evogression.evolution.CreatureEvolutionFittest('y', linear_data, target_num_creatures=10000, num_cycles=3, optimize=5)
        best_creature =  evolution.best_creature
        print('\nBest creature found!')
        print(best_creature)

        predictions = [{'x': i / 2} for i in range(6, 25)]
        predictions = evolution.add_predictions_to_data(predictions, standardized_data=False)
        calculation_x_values = [point['x'] for point in predictions]
        calculated_y_values = [point['y_PREDICTED'] for point in predictions]

        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.plot(calculation_x_values, calculated_y_values, 'g--')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression - Evolution Test')
        plt.show()


if __name__ == '__main__':
    unittest.main()
