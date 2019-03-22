import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
from pprint import pprint as pp
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
        calculated_y_values = [best_creature.calc_target({'x': d['x']}) for d in linear_data]

        print('\nBest creature found!')
        print(f'  linear regression error^2: 50.8')
        print(f'  creature total error^2:    {round(best_error, 1)}')

        print('\nModifiers:')
        pp(best_creature.modifiers)
        print('\n'*2)

        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.scatter([d['x'] for d in linear_data], calculated_y_values)
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
        calcuation_x_values = [i / 2 for i in range(6, 25)]
        calculated_y_values = [best_creature.calc_target({'x': x}) for x in calcuation_x_values]

        print('\nBest creature found!')
        print(f'  linear regression error^2: 50.8')
        print(f'  creature total error^2:    {round(best_error, 1)}')

        print('\nModifiers:')
        pp(best_creature.modifiers)
        print('\n'*2)

        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.scatter(calcuation_x_values, calculated_y_values)
        plt.show()



if __name__ == '__main__':
    unittest.main()
