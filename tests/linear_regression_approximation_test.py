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
    def setUpClass(TestFile):
        TestLinearRegression.creatures =[evogression.EvogressionCreature({'x': None, 'y': None}, 'y') for _ in range(50000)]

    def test_best_creature_linear_regression_1_layer(self):
        best_error = 10000
        best_creature = None
        for cr_index, creature in enumerate(self.creatures):
            error = 0
            for index, x in enumerate(linear_data['x']):
                target_calc = creature.calc_target({'x': x})
                error += abs(target_calc - linear_data['y'][index]) ** 2
            if error < best_error:
                best_error = error
                best_creature = creature
                print(f'New best creature found!  Index: {cr_index}')
        # now have "best_creature"
        calculated_y_values = [best_creature.calc_target({'x': x}) for x in linear_data['x']]

        print('\nBest creature found!')
        print(f'  linear regression error^2: 50.8')
        print(f'  creature total error^2:    {round(best_error, 1)}')

        print('\nModifiers:')
        pp(best_creature.modifiers)
        print('\n'*2)

        plt.scatter(linear_data['x'], linear_data['y'])
        plt.scatter(linear_data['x'], calculated_y_values)
        plt.show()



if __name__ == '__main__':
    unittest.main()
