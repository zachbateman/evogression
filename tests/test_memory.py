import unittest
import cProfile
import sys
sys.path.insert(1, '..')
import evogression
import easy_multip
from test_data import parabola_data
from pprint import pprint as pp
import json



class TestMemory(unittest.TestCase):

    def test_evolution_memory(self):
        options = [
            (500, 5),
            (1000, 5),
            (5000, 5),
            (5000, 10),
            (5000, 15),
            ]
        evolutions = [evogression.evolution.CreatureEvolutionFittest('y', parabola_data, target_num_creatures=t[0], num_cycles=t[1], optimize=False, use_multip=False) for t in options]

        memory_strings = [json.dumps([cr.__dict__ for cr in ev.creatures])
                          + json.dumps(ev.all_data)
                          + json.dumps(ev.all_data_error_sums)
                          + json.dumps([crlist[0].__dict__ for crlist in ev.best_creatures]) for ev in evolutions]

        print('\n\nString sizes of jsoned evolutions:')
        for ms in memory_strings:
            print_memory_size(ms)
        print('\n\n')
        # breakpoint()


def print_memory_size(s):
    print('  100:5  ->  ' + '{:.2E}'.format(len(s)))



if __name__ == '__main__':
    unittest.main()
