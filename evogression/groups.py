'''
Module containing high-level evogression functionality to fit and summarize regressions.
'''
from typing import List, Dict
import random
from collections import defaultdict
import easy_multip
from .evolution import Evolution



def evolution_group(data: list, target_param: str='', num_creatures: int=10000, num_cycles: int=10, group_size: int=4, **kwargs) -> List[Evolution]:
    '''
    Generate a list of fully initialized Evolution objects.
    Any Evolution kwargs may be provided.
    '''
    if 'use_multip' in kwargs:
        del kwargs['use_multip']
        print('Disabling use_multip for Evolution generation in evolution_group.  Using multip for separate Evolution initializations.')
    arg_groups = [(target_param, data, num_creatures, num_cycles, kwargs) for _ in range(group_size)]
    return easy_multip.map(calculate_single_evolution, arg_groups)


def calculate_single_evolution(arg_group: tuple) -> Evolution:
    '''
    Fully initialize and return a single Evolution object.
    Module-level function for arg to easy_multip.
    '''
    target_param, data, num_cr, num_cy, kwargs = arg_group
    return Evolution(target_param, data, num_creatures=num_cr, num_cycles=num_cy, use_multip=False, clear_creatures=True, **kwargs)


def output_group_regression_funcs(group: list):
    '''
    Take list of Evolution objects and output their
    best regressions to a "regression_modules" subdir.
    '''
    for cr_ev in group:
        cr_ev.output_best_regression()


def group_parameter_usage(group: list) -> Dict[str, int]:
    '''
    Combine each Evolution's .parameter_usefulness_count dicts to see which attributes are important.
    '''
    combined_parameter_usefulness = defaultdict(int)
    for cr_ev in group:
        for param, count in cr_ev.parameter_usefulness_count.items():
            combined_parameter_usefulness[param] += count
    return combined_parameter_usefulness


def parameter_pruned_evolution_group(data: list, target_param: str='', max_parameters: int=10, num_creatures: int=10000, num_cycles: int=10, group_size: int=4) -> List[Evolution]:
    '''
    Generate successive groups of Evolution objects and prune least-used
    parameters from the input data each round until only the most useful parameters remain.
    Finally, run a full-blown evolution cycles with the remaining parameters
    for the saved regression modules.

    USE LARGE >>> num_cycles <<< TO INTEGRATE MORE STATISTICALLY VALID PARAMETER USAGE NUMBERS BEFORE REMOVING PARAMETERS
    '''
    if num_creatures < 1000 or num_cycles < 10:
        print('\n> ERROR!  parameter_pruned_evolution_group() can be unstable (infinite loop)\n  if num_creatures and/or num_cycles args are too small.')
        print('  Please use a minimum num_creatures of 1000 and a minimum num_cycles of 10.\n  Higher values num_cycles are encouraged!\n')
        return

    def num_param_to_eliminate(num_extra_param: int) -> int:
        '''Determine how many worst-performing parameters to elimintate in a given round'''
        if num_extra_param > 50:
            return num_extra_param // 2
        elif num_extra_param > 10:
            return num_extra_param // 3
        elif num_extra_param > 3:
            return 2
        elif num_extra_param > 0:
            return 1
        else:
            return 0

    num_parameters = len(data[0].keys()) - 1
    while num_parameters > max_parameters:
        group = evolution_group(data, target_param, num_creatures // 1.6, num_cycles // 1.6, group_size, optimize=False)

        parameter_usage = [(param, count) for param, count in group_parameter_usage(group).items()]
        random.shuffle(parameter_usage)  # so below filter ignores previous order for equally-ranked parameters
        ranked_parameters = sorted(parameter_usage, key=lambda tup: tup[1])
        print(ranked_parameters)

        dead_params = [t[0] for t in ranked_parameters[:num_param_to_eliminate(num_parameters - max_parameters)]]
        for data_point in data:
            for param in dead_params:
                del data_point[param]
        num_parameters = len(data[0].keys()) - 1

    final_group = evolution_group(data, target_param, num_creatures, num_cycles, group_size)
    print('parameter_pruned_evolution_group complete.  Final Parameter usage counts below:')
    for param, count in group_parameter_usage(final_group).items():
        print(f'  {count}: {param}')
    return final_group
