import unittest
import cProfile
import sys
sys.path.insert(1, '..')
import evogression
from data_examples import many_dimension_data


class TestManyDimensionParameterPruningRegression(unittest.TestCase):
    def test_pruned_evolution(self):
        pruned_evolution_group = evogression.parameter_pruned_evolution_group('Target', many_dimension_data, max_parameters=15, creatures=1000, cycles=10, group_size=3)
        pruned_evolution_group.output_param_usage()
        pruned_evolution_group.output_regression()

    def test_parameter_usage_file(self):
        evogression.generate_robust_param_usage_file('Target', many_dimension_data)


if __name__ == '__main__':
    # have to do the below magic to make cProfile work with unittest
    suite = unittest.TestLoader().discover('.')
    def run_tests():
        unittest.TextTestRunner().run(suite)
    cProfile.run('unittest.main()', 'parameter_pruning_test.profile')
