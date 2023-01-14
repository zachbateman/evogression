import unittest
import sys
sys.path.insert(1, '..')
import evogression
from data_examples import linear_data
import matplotlib.pyplot as plt


class TestLinearRegressionEvolution(unittest.TestCase):
    def test_best_creature_evolution(self):
        evolution = evogression.Evolution('y', linear_data, creatures=10000, cycles=3)
        predictions = evolution.predict([{'x': i / 2} for i in range(6, 25)])

        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.plot([point['x'] for point in predictions], [point['y_PREDICTED'] for point in predictions], 'g--')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression - Evolution Test')
        plt.show()


if __name__ == '__main__':
    unittest.main()
