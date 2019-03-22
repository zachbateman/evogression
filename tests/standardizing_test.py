import unittest
import sys
sys.path.insert(1, '..')
import evogression
import statistics
from test_data import parabola_data


class TestDataStandardization(unittest.TestCase):

    def test_linear_data_standardization(self):
        '''
        Test Standardizer class by standardizing parabola_data.
        Check resulting calculations and then revert data to check against original.
        '''
        standardizer = evogression.standardize.Standardizer(parabola_data)
        standardized_data = standardizer.get_standardized_data()

        self.assertTrue(standardizer.data_modifiers['x']['mean'] == 0)
        self.assertTrue(round(standardizer.data_modifiers['x']['stdev'], 3) == 11.979)

        self.assertTrue(round(standardizer.data_modifiers['y']['mean'], 3) == 165.077)
        self.assertTrue(round(standardizer.data_modifiers['y']['stdev'], 3) == 185.392)

        # Check that data is standardized correctly
        self.assertTrue(statistics.mean(d['x'] for d in standardized_data) == 0)
        self.assertTrue(statistics.stdev(d['x'] for d in standardized_data) == 1)
        self.assertTrue(round(statistics.mean(d['y'] for d in standardized_data), 5) == 0)
        self.assertTrue(round(statistics.stdev(d['y'] for d in standardized_data), 5) == 1)

        reverted_values = []
        for d in standardized_data:
            reverted_values.append({'x': standardizer.unstandardize_value('x', d['x']), 'y': standardizer.unstandardize_value('y', d['y'])})

        for i in range(len(parabola_data)):
            self.assertTrue(round(parabola_data[i]['x'], 5) == round(reverted_values[i]['x'], 5))
            self.assertTrue(round(parabola_data[i]['y'], 5) == round(reverted_values[i]['y'], 5))



if __name__ == '__main__':
    unittest.main()
