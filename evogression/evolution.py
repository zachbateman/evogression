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
from pandas import DataFrame
from .creatures import EvogressionCreature
from .standardize import Standardizer
try:
    from .calc_error_sum import calc_error_sum
except ImportError:
    print('\nUnable to import Cython calc_error_sum module!')
    print('If trying to install/run on a Windows computer, you may need to a C compiler.')
    print('See: https://wiki.python.org/moin/WindowsCompilers')
    print('  -> (If running Windows 7, try using Python 3.7 instead of 3.8+)\n')



class BaseEvolution():
    '''
    Creates a framework for evolving groups of creatures.
    This class is designed to be subclassed into more
    specific evolution algorithms.

    Input data must have all numeric values (or None) and
    CANNOT have a column named "N" or "T".
    '''
    def __init__(self,
                 target_parameter: str,
                 all_data: Union[List[Dict[str, float]], DataFrame],
                 num_creatures: int=10000,
                 num_cycles: int=10,
                 force_num_layers: int=0,
                 max_layers: int=10,
                 standardize: bool=True,
                 use_multip: bool=True,
                 fill_none: bool=True,
                 verbose: bool=True,
                 **kwargs) -> None:

        self.target_parameter = target_parameter
        self.standardize = standardize
        self.verbose = verbose

        if type(all_data) == DataFrame:
            self.all_data = all_data.to_dict('records')
        else:
            self.all_data = all_data
        if fill_none:
            self.fill_none_with_median()

        self.num_creatures = int(num_creatures)
        self.num_cycles = int(num_cycles)
        self.force_num_layers = int(force_num_layers)
        self.max_layers = max_layers
        self.use_multip = use_multip

        self.best_creatures: list = []
        self.parameter_usefulness_count: dict=defaultdict(int)

        if self.standardize:
            self.standardizer = Standardizer(self.all_data)
            self.standardized_all_data = self.standardizer.get_standardized_data()
        else:
            self.standardizer = None

        self.data_checks()
        self.creatures = [EvogressionCreature(target_parameter, full_parameter_example=self.all_data[0], layers=force_num_layers, max_layers=max_layers)
                                    for _ in tqdm.tqdm(range(self.num_creatures))]


    def fill_none_with_median(self):
        '''
        Find median value of each input parameter and
        then replace any None values with this median.
        '''
        # Remove any data points that have None for the target/result parameter
        self.all_data = [d for d in self.all_data if d[self.target_parameter] is not None and not math.isnan(d[self.target_parameter])]

        self.param_medians = {}  # used for default parameter values if None is provided in .predict
        is_nan = math.isnan  # local for speed
        parameters_adjusted = []
        for param in self.all_data[0].keys():
            if param != self.target_parameter:
                values = [d[param] for d in self.all_data if d[param] is not None and not is_nan(d[param])]
                if len(values) < len(self.all_data):  # check length so don't have to do below if no replacements
                    median = statistics.median(values) if values else 0.0  # if values is empy list, just set all to zero
                    self.param_medians[param] = median
                    for d in self.all_data:
                        if d[param] is None or is_nan(d[param]):
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

        if 'N' in self.all_data[0]:
            raise InputDataFormatError('ERROR!  Parameter "N" detected in data.  Cannot use "N" or "T" as parameters.')
        if 'T' in self.all_data[0]:
            raise InputDataFormatError('ERROR!  Parameter "T" detected in data.  Cannot use "N" or "T" as parameters.')


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
            if self.verbose:
                self.print_cycle_stats(best_creature=best_creature, error=error, median_error=median_error, best_creature_error=error)

            if self.record_best_creature(best_creature, error) and self.verbose:
                print(f'\n\n\nNEW BEST CREATURE AFTER {counter} ITERATION{"S" if counter > 1 else ""}...')
                print(best_creature)
                print('Total Error: ' + '{0:.2E}'.format(error))

            evolution_cycle_func()


    def evolution_cycle(self):
        '''
        DEFAULT EVOLUTION CYCLE (meant to be replaced in subclasses)
        Run one cycle of evolution that introduces new random creatures,
        kills weak creatures, and mates the remaining ones.
        '''
        # Add random new creatures each cycle (2.0% of num_creatures each time)
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
        self.creatures = self.creatures[:len(self.num_creatures)//2]


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
        print('Median error: ' + '{0:.2E}'.format(median_error))
        print('Best Creature:')
        print(f'  Generation: {best_creature.generation}    Error: ' + '{0:.2E}'.format(error))


    def mate_creatures(self):
        '''Mate creatures to generate new creatures'''
        rand_rand = random.random
        new_creatures = []
        append = new_creatures.append  # local for speed
        self_creatures = self.creatures  # local for speed
        for i in range(0, len(self.creatures), 2):
            if rand_rand() < 0.5:  # only a 50% chance of mating (cuts down on calcs and issues of too many creatures each cycle)
                creature_group = self_creatures[i: i + 2]
                try:
                    new_creature = creature_group[0] + creature_group[1]
                    if new_creature:
                        append(new_creature)
                except IndexError:  # occurs when at the end of self.creatures
                    pass
        self.creatures.extend(new_creatures)


    def optimize_best_creature(self, iterations=30) -> None:
        '''
        Use the creature.mutate_to_new_creature method to
        transform the best_creature into an even better fit.
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

            print('Best error: ' + '{0:.6E}'.format(error))
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


    def output_best_regression(self, output_filename='regression_function', add_error_value=False) -> None:
        '''
        Save this the regression equation/function this evolution has found
        to be the best into a new Python module so that the function itself
        can be imported and used in other code.
        '''
        name_ext = f'___{round(self.best_error, 4)}' if add_error_value else ''

        if self.standardize:
            self.best_creature.output_python_regression_module(output_filename=output_filename, standardizer=self.standardizer, directory='regression_modules', name_ext=name_ext)
        else:
            self.best_creature.output_python_regression_module(output_filename=output_filename, directory='regression_modules', name_ext=name_ext)


    def predict(self, data: Union[Dict[str, float], List[Dict[str, float]], DataFrame], prediction_key: str='', standardized_data: bool=False):
        '''
        Add best_creature predictions to data arg as f'{target}_PREDICTED' new key.
        Return unstandardized dict or list of dicts or DataFrame depending on provided arg.
        '''
        target_param = self.target_parameter  # local variable for speed
        if prediction_key == '':
            prediction_key = f'{target_param}_PREDICTED'

        is_dataframe = True if type(data) == DataFrame else False
        if is_dataframe:
            data = data.to_dict('records')  # will get processed as list

        if isinstance(data, list):  # DataFrames also get processed here (previously converted to a list)
            # make any None values the previously calculated median from the training data
            for d in data:
                for param, val in d.items():
                    if not val:
                        d[param] = self.param_medians.get(param, 0.0)

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
            if is_dataframe:
                unstandardized_data = DataFrame(unstandardized_data)
            return unstandardized_data

        elif isinstance(data, dict):
            # make any None values the previously calculated median from the training data
            for param, val in data.items():
                if not val:
                    data[param] = self.param_medians.get(param, 0.0)

            if not standardized_data and self.standardize:
                data = self.standardizer.convert_parameter_dict_to_standardized(data)
            data[prediction_key] = self.best_creature.calc_target(data)

            if self.standardize:
                unstandardized_data = {}
                for param, value in data.items():
                    unstandardized_data[param] = self.standardizer.unstandardize_value(target_param if param == prediction_key else param, value)
            else:
                unstandardized_data = data
            return unstandardized_data

        else:
            print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')




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

            if kwargs.get('clear_creatures', False):  # save tons of memory when returning object (helps with multiprocessing)
                self.creatures = [self.best_creature]


    def evolution_cycle(self) -> None:
        '''Run one cycle of evolution'''
        self.kill_weak_creatures()
        self.mutate_top_creatures()
        self.mate_creatures()

        # Add random new creatures each cycle to get back to target num creatures
        # Or... cut out extra creatures if have too many (small chance of happening)
        if len(self.creatures) < self.num_creatures:
            self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], layers=self.force_num_layers, max_layers=self.max_layers)
                                           for _ in range(int(round(self.num_creatures - len(self.creatures), 0)))])
        elif len(self.creatures) > self.num_creatures:
            self.creatures = self.creatures[:self.num_creatures]
        random.shuffle(self.creatures)  # used to mix up new creatures in among multip


    def kill_weak_creatures(self) -> None:
        '''Overwrite CreatureEvolution's kill_weak_creatures method'''
        median_error = self.current_median_error  # local for speed
        self.creatures = [creature for creature in self.creatures if creature.error_sum < median_error]


    def mutate_top_creatures(self) -> None:
        '''
        Sprinkle in additional mutated top (~25%) creatures to enhance their behavior.
        And... LIMIT to max 25% (after many iterations, can approach identical arrors and cutoff breaks down.

        Goal is not to just focus on super-optimizing at this point but to encourage growth
        in the better-but still diverse-group of creatures.
        Other randomly-generated creatures still have a chance to become the new best!

        (Ideally, evolution cycles come up with minimally-optimized creatures with best DESIGNED equation...
        ...THEN, the best creature specifically gets optimized at the end)
        '''
        error_cutoff = (self.best_error + self.current_median_error) / 2
        top_mutations = [cr.mutate_to_new_creature() for cr in self.creatures if cr.error_sum < error_cutoff]
        self.creatures.extend(top_mutations[:self.num_creatures//4])




def find_best_creature(creatures: list, target_parameter: str, data: list, standardizer=None, progressbar=True) -> tuple:
    '''
    Determines best EvogressionCreature and returns misc objects (more than otherwise needed)
    so that multiprocessing (easy_multip) can be used.

    This function may be ~50% of the time spent running an Evolution.
    Ugly code here but HEAVILY OPTIMIZED for speed!
    '''
    best_error = 10 ** 190  # outrageously high value to start loop
    errors: list = []
    append_to_errors = errors.append  # local variable for speed
    _calc_error_sum = calc_error_sum  # local for speed
    calculated_creatures: list = []
    append_to_calculated_creatures = calculated_creatures.append  # local variable for speed
    data_length = len(data)

    best_creature = None
    iterable = tqdm.tqdm(creatures) if progressbar else creatures
    actual_target_values = [data_point[target_parameter] for data_point in data]  # pull these values once instead of each time in loop comp below
    for creature in iterable:
        if not creature.error_sum:
            creature.error_sum = _calc_error_sum(creature.calc_target, data, actual_target_values, data_length)
        error = creature.error_sum
        append_to_calculated_creatures(creature)
        append_to_errors(error)
        if error < best_error:
            best_error = error
            best_creature = creature

    median_error = sorted(errors)[len(errors) // 2]  # MEDIAN error
    return (best_creature, best_error, median_error, calculated_creatures)
find_best_creature_multip = easy_multip.decorators.use_multip(find_best_creature)



class InputDataFormatError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return '\n' + str(self.message) if self.message else '\n'
