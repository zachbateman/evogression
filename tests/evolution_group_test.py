import unittest
import sys
sys.path.insert(1, '..')
import evogression
from test_data import linear_data


class TestGroupFunctions(unittest.TestCase):
    def test_evolution_group(self):
        group = evogression.evolution_group(linear_data, target_param='y', group_size=6,
            num_cycles=3, optimize=10, max_layers=2, progressbar=False)
        self.assertTrue(len(group) == 6)
        layer_counts = [evo.best_creature.max_layers for evo in group]
        self.assertTrue(max(layer_counts) <= 2)




if __name__ == '__main__':
    unittest.main(buffer=True)
