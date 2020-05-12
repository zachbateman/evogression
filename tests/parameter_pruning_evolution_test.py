import unittest
import cProfile
import sys
sys.path.insert(1, '..')
import evogression
import easy_multip
from test_data_many_dimensions import data as many_d_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt


class TestManyDimensionParameterPruningRegression(unittest.TestCase):

    def test_best_creature_parabola_regression_evolution(self):
        pruned_evolution_group = evogression.groups.parameter_pruned_evolution_group(many_d_data, target_param='Target', max_parameters=15, num_creatures=1000, num_cycles=10, group_size=3)
        evogression.groups.output_group_regression_funcs(pruned_evolution_group)
        self.assertTrue(True)



if __name__ == '__main__':
    # have to do the below magic to make cProfile work with unittest
    suite = unittest.TestLoader().discover('.')
    def run_tests():
        unittest.TextTestRunner().run(suite)
    cProfile.run('unittest.main()', 'parameter_pruning_test.profile')
