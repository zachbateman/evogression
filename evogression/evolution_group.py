'''
Module containing high-level evogression functionality to fit and summarize regressions.
'''
from typing import List, Dict
import random
import easy_multip
from .evolution import CreatureEvolutionFittest



def evolution_group(data: list, target_param: str='', target_num_creatures: int=10000, num_cycles: int=10, num_groups: int=4, optimize=True, progressbar=True) -> List[CreatureEvolutionFittest]:
    '''
    Generate a list of fully initialized CreatureEvolution objects.
    '''
    arg_groups = [(target_param, data, target_num_creatures, num_cycles, progressbar) for _ in range(num_groups)]
    if optimize:
        return easy_multip.map(calculate_single_evolution, arg_groups)
    else:
        return easy_multip.map(calculate_single_evolution_without_optimization, arg_groups)


def calculate_single_evolution(arg_group: list) -> CreatureEvolutionFittest:
    '''
    Fully initialize and return a single CreatureEvolution object.
    Module-level function for arg to easy_multip.
    '''
    target_param, data, num_cr, num_cy, progressbar = arg_group
    return CreatureEvolutionFittest(target_param, data, target_num_creatures=num_cr, num_cycles=num_cy, use_multip=False, initial_creature_creation_multip=False, optimize='max', progressbar=progressbar, clear_creatures=True)

def calculate_single_evolution_without_optimization(arg_group: list) -> CreatureEvolutionFittest:
    '''
    Fully initialize and return a single CreatureEvolution object.
    Module-level function for arg to easy_multip.
    '''
    target_param, data, num_cr, num_cy, progressbar = arg_group
    return CreatureEvolutionFittest(target_param, data, target_num_creatures=num_cr, num_cycles=num_cy, use_multip=False, initial_creature_creation_multip=False, optimize=False, progressbar=progressbar, clear_creatures=True)


def output_group_regression_funcs(group: list):
    '''
    Take list of CreatureEvolution objects and output their
    best regressions to a "regression_modules" subdir.
    '''
    for cr_ev in group:
        cr_ev.output_best_regression_function_as_module()


def group_parameter_usage(group: list) -> Dict[str, int]:
    '''
    Combine each CreatureEvolution's .parameter_usefulness_count dicts to see which attributes are important.
    '''
    combined_parameter_usefulness = {key: 0 for key in group[0].parameter_usefulness_count}
    for cr_ev in group:
        for param, count in cr_ev.parameter_usefulness_count.items():
            combined_parameter_usefulness[param] += count
    return combined_parameter_usefulness


def parameter_pruned_evolution_group(data: list, target_param: str='', max_parameters: int=10, target_num_creatures: int=10000, num_cycles: int=10, num_groups: int=4) -> List[CreatureEvolutionFittest]:
    '''
    Generate successive groups of CreatureEvolutionFittest objects and prune least-used
    parameters from the input data each round until only the most useful parameters remain.
    Finally, run a full-blown evolution cycles with the remaining parameters
    for the saved regression modules.
    '''
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
        group = evolution_group(data, target_param, target_num_creatures // 2, num_cycles // 2, num_groups, optimize=False)

        parameter_usage = [(param, count) for param, count in group_parameter_usage(group).items()]
        random.shuffle(parameter_usage)  # so below filter ignores previous order for equally-ranked parameters
        ranked_parameters = sorted(parameter_usage, key=lambda tup: tup[1])
        print(ranked_parameters)
        dead_params = [t[0] for t in ranked_parameters[:num_param_to_eliminate(num_parameters - max_parameters)]]
        for data_point in data:
            for param in dead_params:
                del data_point[param]
        num_parameters = len(data[0].keys()) - 1

    final_group = evolution_group(data, target_param, target_num_creatures, num_cycles, num_groups)
    print('parameter_pruned_evolution_group complete.  Final Parameter usage counts below:')
    for param, count in group_parameter_usage(final_group).items():
        print(f'  {count}: {param}')
    return final_group
