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
            (50000, 5),
            (50000, 10),
            (50000, 30),
            ]
        evolutions = [evogression.evolution.CreatureEvolutionFittest('y', parabola_data, target_num_creatures=t[0], num_cycles=t[1], optimize=False, use_multip=True, clear_creatures=True) for t in options]

        memory_strings = [json.dumps([cr.__dict__ for cr in ev.creatures])
                          + json.dumps(ev.all_data)
                          + json.dumps(ev.all_data_error_sums)
                          + json.dumps([crlist[0].__dict__ for crlist in ev.best_creatures]) for ev in evolutions]


        memory_strings = [{'creatures': len(json.dumps([cr.__dict__ for cr in ev.creatures])),
                           'all_data': len(json.dumps(ev.all_data)),
                           'error_sums': len(json.dumps(ev.all_data_error_sums)),
                           'best_creatures': len(json.dumps([crlist[0].__dict__ for crlist in ev.best_creatures]))}
                                for ev in evolutions]
        for i, d in enumerate(memory_strings):
            memory_strings[i]['total'] = sum(d.values())



        print('\n\nString sizes of jsoned evolutions:')
        for option, ms in zip(options, memory_strings):
            print(f'  {option}  ->  ' + '{:.2E}'.format(ms['total']))
            print(f'          creatures: ' + '{:.2E}'.format(ms['creatures']) + f'     ({round(100 * ms["creatures"] / ms["total"], 1)}%)')
            print(f'          all_data: ' + '{:.2E}'.format(ms['all_data']) + f'     ({round(100 * ms["all_data"] / ms["total"], 1)}%)')
            print(f'          error_sums: ' + '{:.2E}'.format(ms['error_sums']) + f'     ({round(100 * ms["error_sums"] / ms["total"], 1)}%)')
            print(f'          best_creatures: ' + '{:.2E}'.format(ms['best_creatures']) + f'     ({round(100 * ms["best_creatures"] / ms["total"], 1)}%)' + '\n')

        print('\n\n')
        breakpoint()




if __name__ == '__main__':
    unittest.main()
