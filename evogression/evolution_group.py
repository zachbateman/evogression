'''
Module containing high-level evogression functionality to fit and summarize regressions.
'''
import random
import easy_multip
from .evolution import CreatureEvolution



def evolution_group(data: list, target_param: str='', target_num_creatures: int=10000, num_cycles: int=10, num_groups: int=4) -> list:
    '''
    Generate a list of fully initialized CreatureEvolution objects.
    '''
    arg_groups = [(target_param, data, target_num_creatures, num_cycles) for _ in range(num_groups)]
    return easy_multip.map(calculate_single_evolution, arg_groups)


def calculate_single_evolution(arg_group):
    '''
    Fully initialize and return a single CreatureEvolution object.
    Module-level function for arg to easy_multip.
    '''
    target_param, data, num_cr, num_cy = arg_group
    return CreatureEvolution(target_param, data, target_num_creatures=num_cr, num_cycles=num_cy, use_multip=False, initial_creature_creation_multip=False)


def output_group_regression_funcs(group: list):
    '''
    Take list of CreatureEvolution objects and output their
    best regressions to a "regression_modules" subdir.
    '''
    for cr_ev in group:
        cr_ev.output_best_regression_function_as_module()


def group_parameter_usage(group: list):
    '''
    Combine each CreatureEvolution's .parameter_usefulness_count dicts to see which attributes are important.
    '''
    combined_parameter_usefulness = {key: 0 for key in group[0].parameter_usefulness_count}
    for cr_ev in group:
        for param, count in cr_ev.parameter_usefulness_count.items():
            combined_parameter_usefulness[param] += count
    return combined_parameter_usefulness
