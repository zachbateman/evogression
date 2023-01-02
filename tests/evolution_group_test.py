import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data
from test_data_many_dimensions import data as many_d_data


class TestGroupFunctions(unittest.TestCase):
    def test_parameter_usage_count(self):
        group = evogression.groups.evolution_group('Target', many_d_data, num_creatures=500, num_cycles=5, group_size=10, optimize=False)
        for ev in group:
            print(ev.parameter_usefulness_count)
            self.assertTrue(len(ev.parameter_usefulness_count) > 0)



if __name__ == '__main__':
    unittest.main(buffer=False)
