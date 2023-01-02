import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
import matplotlib.pyplot as plt


class TestLinearRegressionEvolution(unittest.TestCase):
    def test_best_creature_evolution(self):
        evolution = evogression.Evolution('y', linear_data, num_creatures=10000, num_cycles=3, optimize=5)

        predictions = [{'x': i / 2} for i in range(6, 25)]
        predictions = evolution.predict(predictions)
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
