import unittest
import cProfile
import sys
sys.path.insert(1, '..')
import evogression
import easy_multip
from test_data import parabola_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt


class TestParabolaRegression(unittest.TestCase):

    def test_best_creature_parabola_regression_evolution(self):
        model = evogression.Evolution('y', parabola_data, num_creatures=10000, num_cycles=10, force_num_layers=0, standardize=True)
        model.output_best_regression(directory='regression_modules', add_error_value=True)

        x_values = list(range(-20, 21))
        y_values = [model.predict({'x': x})['y_PREDICTED'] for x in x_values]

        plt.scatter([d['x'] for d in parabola_data], [d['y'] for d in parabola_data])
        plt.plot(x_values, y_values, 'g--')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Parabola Regression - Evolution Test')
        plt.show()



if __name__ == '__main__':
    # have to do the below magic to make cProfile work with unittest
    suite = unittest.TestLoader().discover('.')
    def run_tests():
        unittest.TextTestRunner().run(suite)
    cProfile.run('unittest.main()', 'parabola_test.profile')
