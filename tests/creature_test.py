import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
from pprint import pprint as pp


class TestCreatureCreation(unittest.TestCase):

    @classmethod
    def setUpClass(TestFile):
        TestCreatureCreation.creature = evogression.EvogressionCreature({'x': None, 'y': None}, 'y')

    def test_asdict(self):
        pp(self.creature.get_regression_func())
        self.assertTrue(type(self.creature.modifiers) == dict)

    def test_calc_target(self):
        calculated_values = 0
        for x in linear_data['x']:
            calculated_values += self.creature.calc_target({'x': x})
        self.assertTrue(type(calculated_values) == float)



if __name__ == '__main__':
    unittest.main()
