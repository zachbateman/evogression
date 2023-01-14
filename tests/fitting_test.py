import unittest
import sys
sys.path.insert(1, '..')
import evogression
from data_examples import linear_data, parabola_data
import matplotlib.pyplot as plt


class Test2DRegression(unittest.TestCase):
    def test_linear_regression(self):
        evolution = evogression.Evolution('y', linear_data, creatures=10000, cycles=3)
        predictions = evolution.predict([{'x': i / 2} for i in range(6, 25)])

        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])
        plt.plot([point['x'] for point in predictions], [point['y_PREDICTED'] for point in predictions], 'g--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression - Evolution Test')
        plt.show()
        
    def test_parabola_regression(self):
        model = evogression.Evolution('y', parabola_data, creatures=10000, cycles=10)
        model.output_regression(directory='regression_modules', add_error_value=True)
        x_values = list(range(-20, 21))
        y_values = [model.predict({'x': x})['y_PREDICTED'] for x in x_values]

        plt.scatter([d['x'] for d in parabola_data], [d['y'] for d in parabola_data])
        plt.plot(x_values, y_values, 'g--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Parabola Regression - Evolution Test')
        plt.show()


if __name__ == '__main__':
    unittest.main()
