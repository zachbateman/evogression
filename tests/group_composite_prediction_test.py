import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
import matplotlib.pyplot as plt


class TestGroupComposite(unittest.TestCase):
    def test_evolution_group(self):
        group = evogression.evolution_group('y', linear_data, num_creatures=1000, group_size=50, num_cycles=7)

        calculation_x_values = [i / 2 for i in range(6, 27)]
        for evo in group:
            calculated_y_values = [evo.predict({'x': x}, 'pred')['pred'] for x in calculation_x_values]
            plt.plot(calculation_x_values, calculated_y_values, alpha=0.1)

        calculated_y_values = [sum(evo.predict({'x': x}, 'pred')['pred'] for evo in group) / len(group) for x in calculation_x_values]
        plt.plot(calculation_x_values, calculated_y_values,  'g--', alpha=1.0)
        plt.scatter([d['x'] for d in linear_data], [d['y'] for d in linear_data])

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression - Group Composite Prediction Test')
        plt.show()



if __name__ == '__main__':
    unittest.main(buffer=True)
