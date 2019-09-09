import unittest
import os
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
from pprint import pprint as pp


class TestCreatureCreation(unittest.TestCase):

    @classmethod
    def setUpClass(TestFile):
        TestCreatureCreation.creature = evogression.EvogressionCreature('y', full_parameter_example={'x': None, 'y': None})
        TestCreatureCreation.standardizer = evogression.standardize.Standardizer(linear_data)

    def test_asdict(self):
        pp(self.creature.get_regression_func())
        self.assertTrue(type(self.creature.modifiers) == dict)

    def test_calc_target(self):
        calculated_values = 0
        for d in linear_data:
            calculated_values += self.creature.calc_target(d)
        self.assertTrue(type(calculated_values) == float)

    def test_python_module_regression_func_output(self):
        self.creature.output_python_regression_module(output_filename='test_reg_func.py')
        self.assertTrue(os.path.exists('test_reg_func.py'))
        os.remove('test_reg_func.py')

    def test_python_module_regression_func_output_with_standardizer(self):
        self.creature.output_python_regression_module(output_filename='test_reg_func_with_standardizer.py', standardizer=self.standardizer)
        self.assertTrue(os.path.exists('test_reg_func_with_standardizer.py'))
        os.remove('test_reg_func_with_standardizer.py')

    def test_creature_addition(self):
        creatures = [evogression.EvogressionCreature('y', full_parameter_example={'x': None, 'y': None}) for _ in range(1000)]
        new_creatures = []
        for i in range(0, len(creatures), 2):
            creature_group = creatures[i: i + 2]
            try:
                new_creature = creature_group[0] + creature_group[1]
                if new_creature:
                    new_creatures.append(new_creature)
            except IndexError:
                pass
        creatures.extend(new_creatures)
        for cr in creatures:
            for i in range(1, len(cr.modifiers) + 1):
                if i > 1:
                    self.assertTrue('T' in cr.modifiers[f'LAYER_{i}'])

    def test_creature_mutation(self):
        start_gen = self.creature.generation
        new_creature = self.creature.mutate_to_new_creature()
        end_gen = new_creature.generation
        self.assertTrue(start_gen != end_gen)
        self.assertTrue(self.creature.modifiers != new_creature.modifiers)



if __name__ == '__main__':
    unittest.main(buffer=True)
