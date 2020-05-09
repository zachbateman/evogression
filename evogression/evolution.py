'''
Module containing evolution algorithms for regression.
'''
from typing import List, Dict, Union
import statistics
import copy
import random
import math
import tqdm
import warnings
from collections import defaultdict
import easy_multip
from pprint import pprint as pp
from . import creatures
from .creatures import EvogressionCreature
from .standardize import Standardizer



class BaseEvolution():
    '''
    Creates a framework for evolving groups of creatures.
    This class is designed to be subclassed into more
    specific evolution algorithms.
    '''
    def __init__(self,
                 target_parameter: str,
                 all_data: List[Dict[str, float]],
                 num_creatures: int=10000,
                 add_random_creatures_each_cycle: bool=True,
                 num_cycles: int=10,
                 force_num_layers: int=0,
                 max_layers=None,
                 standardize: bool=True,
                 use_multip: bool=True,
                 fill_none: bool=True,
                 **kwargs) -> None:

        self.target_parameter = target_parameter
        self.standardize = standardize
        self.all_data = all_data
        if fill_none:
            self.fill_none_with_median()

        self.num_creatures = num_creatures
        self.add_random_creatures_each_cycle = add_random_creatures_each_cycle
        self.num_cycles = num_cycles
        self.force_num_layers = force_num_layers
        self.max_layers = max_layers
        # could reset module-level layer_probabilities list used by creatues to not have more than max_layers as options
        # creatures.layer_probabilities = [n for n in creatures.layer_probabilities if n <= max_layers]
        self.use_multip = use_multip

        self.current_generation = 1
        self.best_creatures: list = []
        self.parameter_usefulness_count: dict=defaultdict(int)

        if self.standardize:
            self.standardizer = Standardizer(self.all_data)
            self.standardized_all_data = self.standardizer.get_standardized_data()
        else:
            self.standardizer = None

        self.data_checks()
        self.creatures = [EvogressionCreature(target_parameter, full_parameter_example=self.all_data[0], layers=force_num_layers, max_layers=max_layers)
                                    for _ in tqdm.tqdm(range(num_creatures))]


    def fill_none_with_median(self):
        '''
        Find median value of each input parameter and
        then replace any None values with this median.
        '''
        # Remove any data points that have None for the target/result parameter
        self.all_data = [d for d in self.all_data if d[self.target_parameter] is not None and not math.isnan(d[self.target_parameter])]

        parameters_adjusted = []
        for param in self.all_data[0].keys():
            if param != self.target_parameter:
                values = [d[param] for d in self.all_data if d[param] is not None and not math.isnan(d[param])]
                if len(values) < len(self.all_data):  # check length so don't have to do below if no replacements
                    median = statistics.median(values)
                    for d in self.all_data:
                        if d[param] is None or math.isnan(d[param]):
                            parameters_adjusted.append(param)
                            d[param] = median

        if len(parameters_adjusted) >= 1:
            print('Data None values filled with median for the following parameters:')
            for param in sorted(set(parameters_adjusted)):
                print(f'  {param}')


    def data_checks(self):
        '''
        Check cleaned input data for potential issues.
        At the point when this is called, there should be no issues with
        the data to be used; data-cleaning methods are called earlier.

        If this method prints errors, we need to write more data-cleaning capabilities to handle those cases!
        '''
        acceptable_types = {'float', 'int', 'float64', 'int64'}
        issues = []
        def check_data(data, data_name):
            for i, d in enumerate(data):
                for key, val in d.items():
                    # val > & < checks are way of checking for nan without needing to require numpy import
                    if type(val).__name__ not in acceptable_types or not (val >= 0 or val <= 0):
                        issues.append((data_name, i, key, val))

        check_data(self.all_data, 'all_data')
        check_data(self.standardized_all_data, 'standardized_all_data')
        for issue in issues:
            data_name, i, key, val = issue
            print(f'\nERROR!  NAN values detected in {data_name}!')
            print(f'Index: {i}  key: {key}  value: {val}  type: {type(val).__name__}')


    def additional_best_creatures(self) -> list:
        '''
        Sprinkle in additional mutated best_creatures to enhance this behavior.

        DON'T WANT VERY MANY as the goal is not to just focus on super-optimizing the
        best creature found from the first round of evolution.
        Instead, add a FEW mutations of any given best_creature so that other randomly-generated
        creatures still have a chance to become the new best!

        (Ideally, evolution cycles come up with minimally-optimized creature with best DESIGNED equation...
        ...THEN, that creature specifically gets optimized at the end)
        '''
        return [self.best_creature.mutate_to_new_creature() for _ in range(5)]


    def evolve_creatures(self, evolution_cycle_func=None, progressbar=True):
        '''
        Main evolution loop that handles results of each loop and
        keeps track of best creatures/regression equations.
        '''
        if evolution_cycle_func is None:
            evolution_cycle_func = self.evolution_cycle

        counter = 0
        while counter < self.num_cycles:
            counter += 1
            print('----------------------------------------' + f'\nCycle - {counter} -')

            best_creature, error, median_error = self.calculate_all_and_find_best_creature(progressbar=progressbar)
            self.current_median_error = median_error
            self.print_cycle_stats(best_creature=best_creature, error=error, median_error=median_error, best_creature_error=error)

            if self.record_best_creature(best_creature, error):
                print(f'\n\n\nNEW BEST CREATURE AFTER {counter} ITERATION{"S" if counter > 1 else ""}...')
                print(best_creature)
                print('Total Error: ' + '{0:.2E}'.format(error))

            self.creatures.extend(self.additional_best_creatures())  # sprinkle in additional best_creature mutants
            evolution_cycle_func()


    def evolution_cycle(self):
        '''
        Default evolution cycle.
        Run one cycle of evolution that introduces new random creatures,
        kills weak creatures, and mates the remaining ones.
        '''
        # Option to add random new creatures each cycle (2.0% of num_creatures each time)
        if self.add_random_creatures_each_cycle:
            self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], layers=self.force_num_layers, max_layers=self.max_layers) for _ in range(int(round(0.02 * self.num_creatures, 0)))])

        random.shuffle(self.creatures)  # used to mix up new creatures in among multip and randomize feeding groups
        self.kill_weak_creatures()
        self.mate_creatures()


    def record_best_creature(self, best_creature, error) -> bool:
        '''
        Saves a copy of the provided best creature (and its error)
        into self.best_creatures list.

        Returns True/False depending on if the recorded creature is a new
        BEST creature (compared to previously recorded best creatures).

        Also record parameters used in regression equation (modifiers dict)
        to parameter_usefulness_count if a new best error/creatures.
        '''
        new_best_creature = False
        if error < self.best_error:
            new_best_creature = True
            # now count parameter usage if better than previous best creatures
            for param in best_creature.used_parameters():
                self.parameter_usefulness_count[param] += 1

        self.best_creatures.append([copy.deepcopy(best_creature), error])
        return new_best_creature


    @property
    def best_creature(self) -> EvogressionCreature:
        '''
        Return the best creature available in the self.best_creatures list.
        '''
        best_creature, best_error = None, 10 ** 150
        for creature_list in self.best_creatures:
            if creature_list[1] < best_error:
                best_error = creature_list[1]
                best_creature = creature_list[0]
        return best_creature

    @property
    def best_error(self) -> float:
        '''
        Return error associated with best creature available in self.best_creatures list.
        If no existing best_creatures, return default huge error.
        '''
        best_error = 10 ** 150
        for creature_list in self.best_creatures:
            if creature_list[1] < best_error:
                best_error = creature_list[1]
        return best_error


    def kill_weak_creatures(self):
        '''Remove half of the creatures randomly (self.creatures was previously shuffled)'''
        self.creatures = self.creatures[:len(self.creatures)//2]


    def stats_from_find_best_creature_multip_result(self, result_data: list) -> tuple:
        '''
        Unpack and return metrics from the data provided by the multip version of find_best_creature.

        result_data arg is a list where each 4 items is the output from one find_best_creature call.
        Specifically, result_data looks like:
            [EvogressionCreature, best_error, avg_error, calculated_creatures,
              EvogressionCreature, ...,
              ...
            ]
        '''
        calculated_creatures = []
        best_creature_lists = [result_data[4 * i: 4 * (i + 1)] for i in range(int(len(result_data) / 4))]  # make result_data useful
        # best_creature_lists is list with items of form [best_creature, error, avg_error]
        error, best_creature = None, None
        for index, bc_list in enumerate(best_creature_lists):
            calculated_creatures.extend(bc_list[3])
            if error is None or bc_list[1] < error:
                error = bc_list[1]
                best_creature = bc_list[0]
        median_error = sum(bc_list[2] for bc_list in best_creature_lists) / len(best_creature_lists)  # mean of medians of big chunks...
        return best_creature, error, median_error, calculated_creatures


    def calculate_all_and_find_best_creature(self, progressbar=True) -> tuple:
        '''
        Find the best creature in all current creatures by calculating each one's
        total error compared to all the training data.
        '''
        if self.use_multip:
            if self.standardize:
                result_data = find_best_creature_multip(self.creatures, self.target_parameter,
                                                                             self.standardized_all_data, standardizer=self.standardizer,
                                                                             progressbar=progressbar)
            else:
                result_data = find_best_creature_multip(self.creatures, self.target_parameter,
                                                                              self.all_data, progressbar=progressbar)
            best_creature, error, median_error, calculated_creatures = self.stats_from_find_best_creature_multip_result(result_data)
        else:
            best_creature, error, median_error, calculated_creatures = find_best_creature(self.creatures, self.target_parameter, self.standardized_all_data, progressbar=progressbar)
        self.creatures = calculated_creatures
        return best_creature, error, median_error


    def print_cycle_stats(self, best_creature=None, error=None, median_error: float=0, best_creature_error: float=0) -> None:
        print(f'Total number of creatures:  {len(self.creatures)}')
        print('Median error: ' + '{0:.2E}'.format(median_error))
        print('Best Creature:')
        print(f'  Generation: {best_creature.generation}    Error: ' + '{0:.2E}'.format(error))


    def mate_creatures(self):
        '''Mate creatures to generate new creatures'''
        new_creatures = []
        new_creatures_append = new_creatures.append  # local var for speed
        for i in range(0, len(self.creatures), 2):
            creature_group = self.creatures[i: i + 2]
            try:
                new_creature = creature_group[0] + creature_group[1]
                if new_creature:
                    new_creatures_append(new_creature)
            except IndexError:  # occurs when at the end of self.creatures
                pass
        self.creatures.extend(new_creatures)


    def optimize_best_creature(self, iterations=30):
        '''
        Use the creature.mutate_to_new_creature method to transform
        the best_creature into an even better fit.
        '''
        print('\n\n\nOptimizing best creature...')
        best_creature = self.best_creature
        print(best_creature)
        errors = []
        adjustments = 'fast'  # start out with larger, faster mutations
        for i in tqdm.tqdm(range(iterations)):
            if i > iterations / 3:  # quickly mutate creature for first 1/3rd of iterations and then make small, fine mutations
                adjustments = 'fine'

            mutated_clones = [best_creature] + [best_creature.mutate_to_new_creature(adjustments=adjustments) for _ in range(500)]

            if self.use_multip:
                result_data = find_best_creature_multip(mutated_clones, self.target_parameter, self.standardized_all_data, progressbar=False)
                best_creature, error, median_error, calculated_creatures = self.stats_from_find_best_creature_multip_result(result_data)
            else:
                best_creature, error, median_error, calculated_creatures = find_best_creature(mutated_clones, self.target_parameter, self.standardized_all_data, progressbar=False)

            print(f'Best error: ' + '{0:.6E}'.format(error))
            errors.append(error)
            if error == 0:
                break  # break out of loop if no error/perfect regression
            if i > 5 and error / errors[-3] > 0.9999:
                if adjustments == 'fast':
                    adjustments = 'fine'
                else:
                    break  # break out of the loop if it's no longer improving accuracy
        self.record_best_creature(best_creature, error)
        print(self.best_creature)
        print('Best creature optimized!\n')


    def output_best_regression(self, output_filename='regression_function', add_error_value=False):
        '''
        Save this the regression equation/function this evolution has found
        to be the best into a new Python module so that the function itself
        can be imported and used in other code.
        '''
        if add_error_value:
            name_ext = f'___{round(self.best_error, 4)}'
        else:
            name_ext = ''

        if self.standardize:
            self.best_creature.output_python_regression_module(output_filename=output_filename, standardizer=self.standardizer, directory='regression_modules', name_ext=name_ext)
        else:
            self.best_creature.output_python_regression_module(output_filename=output_filename, directory='regression_modules', name_ext=name_ext)


    def add_predictions_to_data(self, data: List[Dict[str, float]], standardized_data: bool=False) -> List[dict]:
        '''
        DEPRECATED AND WILL BE REMOVED IN FAVOR OF .predict()

        Add best_creature predictions to data arg as f'{target}_PREDICTED' new key.
        Return unstandardized list of dicts.
        '''
        warnings.warn('.add_predictions_to_data() will be removed; please use .predict() instead', DeprecationWarning)

        pred_key = f'{self.target_parameter}_PREDICTED'
        none_counter = 0
        target_param = self.target_parameter  # local variable for speed
        for d in data:
            if target_param in d and d[target_param] is None:
                d[target_param] = -99999
                none_counter += 1
        if none_counter > 0:
            print('\nWhile adding predictions to provided data set,\n  None values were found in the target parameter.')
            print(f'  {none_counter} target parameter None values were replaced with -99999\n')

        if not standardized_data and self.standardize:
            data = [self.standardizer.convert_parameter_dict_to_standardized(d) for d in data]

        for d in data:
            d[pred_key] = self.best_creature.calc_target(d)

        if self.standardize:
            unstandardized_data = []
            for d in data:
                unstandardized_d = {}
                for param, value in d.items():
                    unstandardized_d[param] = self.standardizer.unstandardize_value(target_param if '_PREDICTED' in param else param, value)
                unstandardized_data.append(unstandardized_d)
        else:
            unstandardized_data = data

        return unstandardized_data


    def predict(self, data: Union[Dict[str, float], List[Dict[str, float]]], standardized_data: bool=False, prediction_key: str=''):
        '''
        Add best_creature predictions to data arg as f'{target}_PREDICTED' new key.
        Return unstandardized dict or list of dicts depending on provided arg.
        '''
        target_param = self.target_parameter  # local variable for speed
        if prediction_key == '':
            prediction_key = f'{target_param}_PREDICTED'

        if type(data) == list:
            if not standardized_data and self.standardize:
                data = [self.standardizer.convert_parameter_dict_to_standardized(d) for d in data]
            for d in data:
                d[prediction_key] = self.best_creature.calc_target(d)

            if self.standardize:
                unstandardized_data = []
                for d in data:
                    unstandardized_d = {}
                    for param, value in d.items():
                        unstandardized_d[param] = self.standardizer.unstandardize_value(target_param if param == prediction_key else param, value)
                    unstandardized_data.append(unstandardized_d)
            else:
                unstandardized_data = data
            return unstandardized_data

        elif type(data) == dict:
            if not standardized_data and self.standardize:
                data = self.standardizer.convert_parameter_dict_to_standardized(data)
            data[prediction_key] = self.best_creature.calc_target(data)

            if self.standardize:
                unstandardized_data = {}
                for param, value in data.items():
                    unstandardized_data[param] = self.standardizer.unstandardize_value(target_param if '_PREDICTED' in param else param, value)
            else:
                unstandardized_data = data
            return unstandardized_data

        else:
            print('Error!  "data" arg provided to .predict() must be a dict or list of dicts.')




class Evolution(BaseEvolution):
    '''
    MAIN/REFERENCE EVOLUTION ALOGORITHM

    Evolves creatures by killing off the worst performers in
    each cycle and then randomly generating many new creatures.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.evolve_creatures(self.evolution_cycle, progressbar=kwargs.get('progressbar', True))
            optimize = kwargs.get('optimize', True)
            if optimize == 'max':
                self.optimize_best_creature(iterations=100)
            elif type(optimize) == int:
                if optimize > 0:
                    self.optimize_best_creature(iterations=optimize)
                else:
                    print('Warning!  Optimization cycles must be an int > 0!')
            elif optimize:
                self.optimize_best_creature()

            if kwargs.get('clear_creatures', False):  # save tons of memory when returning object (helps with multip)
                self.creatures = [self.best_creature]


    def evolution_cycle(self):
        '''Run one cycle of evolution'''
        self.kill_weak_creatures()
        self.mate_creatures()

        # Add random new creatures each cycle to get back to target num creatures
        self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], layers=self.force_num_layers, max_layers=self.max_layers) for _ in range(int(round(self.num_creatures - len(self.creatures), 0)))])
        random.shuffle(self.creatures)  # used to mix up new creatures in among multip


    def kill_weak_creatures(self):
        '''Overwrite CreatureEvolution's kill_weak_creatures method'''
        median_error = self.current_median_error  # local for speed
        self.creatures = [creature for creature in self.creatures if creature.error_sum < median_error]







def calc_error_value(creature, target_parameter: str, data_point: dict, standardizer=None) -> float:
    '''
    Calculate the error between a creature's predicted value and the actual value.
    data_point must ALREADY be standardized if using a standardizer!!!
    '''
    target_calc = creature.calc_target(data_point)
    data_point_calc = data_point[target_parameter]
    if standardizer is not None:
        unstandardize_value = standardizer.unstandardize_value
        target_calc = unstandardize_value(target_parameter, target_calc)
        data_point_calc = unstandardize_value(target_parameter, data_point_calc)
    try:
        error = (target_calc - data_point_calc) ** 2.0  # sometimes generates "RuntimeWarning: overflow encountered in double_scalars"
    except OverflowError:  # if error is too big to store, give huge arbitrary error
        error = 10 ** 150
    return error


def find_best_creature(creatures: list, target_parameter: str, data: list, standardizer=None, progressbar=True) -> tuple:
    '''
    Determines best EvogressionCreature and returns misc objects (more than otherwise needed)
    so that multiprocessing (easy_multip) can be used.
    '''
    best_error = -1  # to start loop
    errors: list = []
    append_to_errors = errors.append  # local variable for speed
    calculated_creatures: list = []
    append_to_calculated_creatures = calculated_creatures.append  # local variable for speed
    data_length = len(data)
    best_creature = None
    iterable = tqdm.tqdm(creatures) if progressbar else creatures
    for creature in iterable:
        if not creature.error_sum:
            creature.error_sum = sum([calc_error_value(creature, target_parameter, data_point, standardizer) for data_point in data]) / data_length
        error = creature.error_sum

        append_to_calculated_creatures(creature)
        if error < best_error or best_error < 0:
            best_error = error
            best_creature = creature
        append_to_errors(error)
    avg_error = sorted(errors)[len(errors) // 2]  # MEDIAN error
    return (best_creature, best_error, avg_error, calculated_creatures)
find_best_creature_multip = easy_multip.decorators.use_multip(find_best_creature)
