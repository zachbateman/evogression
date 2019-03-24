import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import parabola_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt


class TestParabolaRegression(unittest.TestCase):

    def test_best_creature_parabola_regression_brute_force_3_layer(self):
        # this method (creating random, independent creatures and picking the best one)
        # is generating a great match with 100,000 creatures and squared error calculations!!!
        creatures = [evogression.EvogressionCreature('y', full_parameter_example={'x': None, 'y': None}, layers=3) for _ in range(100000)]

        best_error = -1
        best_creature = None
        for cr_index, creature in enumerate(creatures):
            error = 0
            for d in parabola_data:
                target_calc = creature.calc_target({'x': d['x']})
                error += abs(target_calc - d['y']) ** 2
            if error < best_error or best_error < 0:
                best_error = error
                best_creature = creature
                print(f'New best creature found!  Index: {cr_index}')
        # now have "best_creature"
        calculation_x_values = [i for i in range(-20, 21)]
        calculated_y_values = [best_creature.calc_target({'x': x}) for x in calculation_x_values]

        print('\nBest creature found!')
        print(f'  creature total error^2:    {round(best_error, 1)}')

        print('\nModifiers:')
        pp(best_creature.modifiers)
        print('\n'*2)

        plt.scatter([d['x'] for d in parabola_data], [d['y'] for d in parabola_data])
        plt.scatter(calculation_x_values, calculated_y_values)
        plt.show()


    def test_best_creature_parabola_regression_evolution(self):
        '''
        HAVING A VERY HARD TIME GETTING A GOOD PARABOLA MATCH WITH EVOLUTION!!!
        TRY GETTING EVOLUTION AS CLOSE AS POSSIBLE TO THE STEPS IN BRUTE FORCE ABOVE BEFORE GETTING TOO CLEVER!!!
        '''
        evolution = evogression.evolution.CreatureEvolution('y', parabola_data, target_num_creatures=50000, num_cycles=7, force_num_layers=0, standardize=True)
        try:
            best_creature, standardizer = evolution.return_best_creature()
        except:
            best_creature = evolution.return_best_creature()

        calculated_y_values = []
        for d in parabola_data:
            try:
                standardized_dict = standardizer.convert_parameter_dict_to_standardized(d)
                standardized_value = best_creature.calc_target(standardized_dict)
                calculated_y_values.append(standardizer.unstandardize_value('y', standardized_value))
            except:
                value = best_creature.calc_target(d)
                calculated_y_values.append(value)

        plt.scatter([d['x'] for d in parabola_data], [d['y'] for d in parabola_data])
        plt.scatter([d['x'] for d in parabola_data], calculated_y_values)
        plt.show()



if __name__ == '__main__':
    unittest.main()
