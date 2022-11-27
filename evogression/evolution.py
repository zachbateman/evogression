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
import pickle
from pandas import DataFrame
from .creatures import EvogressionCreature
from .standardize import Standardizer
from . import data
try:
    from .calc_error_sum import calc_error_sum
except ImportError:
    print('\nUnable to import Cython calc_error_sum module!')
    print('If trying to install/run on a Windows computer, you may need to a C compiler.')
    print('See: https://wiki.python.org/moin/WindowsCompilers')
    print('  -> (If running Windows 7, try using Python 3.7 instead of 3.8+)\n')

from . import rust_evogression


# function must be module-level so that it is pickleable for multiprocessing
# making it a global variable that is a function set in evolution __init__ to be module-level but still customizable
find_best_creature_multip = None


class Evolution():
    '''
    EVOLUTION ALOGORITHM

    Evolves creatures by killing off the worst performers in
    each cycle and then randomly generating many new creatures.

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
                 num_cpu: int=1,
                 verbose: bool=True,
                 optimize = True,
                 **kwargs) -> None:

        self.target_parameter = target_parameter
        self.verbose = verbose

        if type(all_data) == DataFrame:
            all_data = all_data.to_dict('records')
        else:
            all_data = all_data

        data.data_checks(all_data)

        self.param_medians = data.calc_param_medians(all_data, target_parameter)
        self.all_data = data.fill_none_with_median(all_data, target_parameter, self.param_medians)

        self.num_creatures = int(num_creatures)
        self.num_cycles = int(num_cycles)
        self.force_num_layers = int(force_num_layers)
        self.max_layers = int(max_layers)
        self.num_cpu = num_cpu if num_cpu >= 1 else 1
        global find_best_creature_multip
        find_best_creature_multip = easy_multip.decorators.use_multip(find_best_creature, num_cpu=self.num_cpu)

        import time
        t1_a = time.time()
        evo_data = rust_evogression.run_evolution(target_parameter, self.all_data, num_creatures, num_cycles, max_layers)
        # standerdizer_dicts, best_creatures, best_creature = evo_data

        from pprint import pprint as pp
        pp(evo_data)
        std, best_cr = evo_data
        t1_b = time.time()

        
        self.best_creatures: list = []
        self.parameter_usefulness_count: dict = defaultdict(int)
        
        t2_a = time.time()
        self.standardizer = Standardizer(self.all_data)
        self.standardized_all_data = self.standardizer.get_standardized_data()

        self.creatures = [EvogressionCreature(target_parameter, full_parameter_example=self.all_data[0], layers=force_num_layers, max_layers=max_layers)
                                    for _ in tqdm.tqdm(range(self.num_creatures))]


        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            self.evolve_creatures(progressbar=kwargs.get('progressbar', True))

            if optimize == 'max':
                self.optimize_best_creature(iterations=100)
            elif isinstance(optimize, int):
                if optimize > 0:
                    self.optimize_best_creature(iterations=optimize)
                else:
                    print('Warning!  Optimization cycles must be an int > 0!')
            elif optimize:
                self.optimize_best_creature()

            # save tons of memory when returning object (helps with multiprocessing)
            self.creatures = [self.best_creature]  
        t2_b = time.time()

        print(f'Time for Rust version: {t1_b - t1_a:,.3f} seconds')
        print(f'Time for Python version: {t2_b - t2_a:,.3f} seconds')


    def evolve_creatures(self, progressbar=True) -> None:
        '''
        Main evolution loop that handles results of each loop and
        keeps track of best creatures/regression equations.
        '''
        counter = 0
        while counter < self.num_cycles:
            counter += 1
            if self.verbose:
                print('----------------------------------------' + f'\nCycle - {counter} -')

            best_creature, error, median_error = self.calculate_all_and_find_best_creature(progressbar=progressbar)
            self.current_median_error = median_error
            if self.verbose:
                self.print_cycle_stats(best_creature=best_creature, error=error, median_error=median_error, best_creature_error=error)

            if self.record_best_creature(best_creature, error) and self.verbose:
                print(f'\n\nNEW BEST CREATURE AFTER {counter} ITERATION{"S" if counter > 1 else ""}...')
                print(best_creature)
                print('Total Error: ' + '{0:.2E}'.format(error))

            self.evolution_cycle()


    def evolution_cycle(self) -> None:
        '''
        Run one cycle of evolution that introduces new random creatures,
        kills weak creatures, and mates the remaining ones.
        '''
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


    def kill_weak_creatures(self) -> None:
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
        if self.num_cpu > 1:
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
        print('\rMedian error: ' + '{0:.2E}'.format(median_error))
        print('Best Creature:')
        print(f'  Generation: {best_creature.generation}    Error: ' + '{0:.2E}'.format(error))


    def mate_creatures(self) -> None:
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
        best_creature = self.best_creature
        if self.verbose:
            print('\n\n\nOptimizing best creature...')
            print(best_creature)
        errors = []
        adjustments = 'fast'  # start out with larger, faster mutations
        with tqdm.tqdm(total=iterations, position=1, bar_format='{desc}', desc='...') as desc:
            for i in tqdm.tqdm(range(iterations), desc='Optimizing', position=0, leave=True):
                if i > iterations / 3:  # quickly mutate creature for first 1/3rd of iterations and then make small, fine mutations
                    adjustments = 'fine'

                mutated_clones = [best_creature] + [best_creature.mutate_to_new_creature(adjustments=adjustments) for _ in range(500)]

                if self.num_cpu > 1:
                    result_data = find_best_creature_multip(mutated_clones, self.target_parameter, self.standardized_all_data, progressbar=False)
                    best_creature, error, median_error, calculated_creatures = self.stats_from_find_best_creature_multip_result(result_data)
                else:
                    best_creature, error, median_error, calculated_creatures = find_best_creature(mutated_clones, self.target_parameter, self.standardized_all_data, progressbar=False)

                errors.append(error)
                if self.verbose:
                    # print('\nBest error: ' + '{0:.6E}'.format(error))
                    desc.set_description(f'Start: {errors[0]:.5E}     Best: {error:.5E}')

                if error == 0:
                    break  # break out of loop if no error/perfect regression
                if i > 5 and error / errors[-3] > 0.9999:
                    if adjustments == 'fast':
                        adjustments = 'fine'
                    else:
                        break  # break out of the loop if it's no longer improving accuracy
        self.record_best_creature(best_creature, error)
        print('\nBest creature optimized!')
        print(self.best_creature)


    def clear_data(self) -> None:
        '''
        Clear out references to data to shrink full object and
        limit memory growth when generating many Evolutions.
        '''
        self.all_data = None
        self.creatures = None
        self.standardizer.all_data = None
        self.standardizer.standardized_data = None


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


    def save(self, filename='evolution_model') -> None:
        '''
        Save this Evolution Python object/model in a pickle file.
        Also removes non-essential data to reduce file size.
        '''
        # Clearing or shrinking these attributes provides a smaller file size.
        self.best_creatures[-10:]
        self.creatures = [self.best_creature]
        self.all_data = None
        self.standardized_all_data = None
        self.standardizer.all_data = None
        self.standardizer.standardized_data = None

        with open(filename if '.' in filename else filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, filename):
        '''
        Load an Evolution object from a saved, pickle file.
        '''
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except:
            with open(filename + '.pkl', 'rb') as f:
                return pickle.load(f)


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
            parameter_example = self.best_creature.full_parameter_example
            for d in data:
                # errors result if leave in key:values not used in training (string split categories for example), so next line ensures minimum data is fed to .calc_target
                clean_d = {key: value for key, value in d.items() if key in parameter_example}
                d[prediction_key] = self.best_creature.calc_target(clean_d)

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
            # errors result if leave in key:values not used in training (string split categories for example), so next line ensures minimum data is fed to .calc_target
            clean_data = {key: value for key, value in data.items() if key in self.best_creature.full_parameter_example}
            data[prediction_key] = self.best_creature.calc_target(clean_data)

            if self.standardize:
                unstandardized_data = {}
                for param, value in data.items():
                    unstandardized_data[param] = self.standardizer.unstandardize_value(target_param if param == prediction_key else param, value)
            else:
                unstandardized_data = data
            return unstandardized_data

        else:
            print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')


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




def find_best_creature(creatures: list, target_parameter: str, data: list, standardizer=None, progressbar=True, cpu_index=0) -> tuple:
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
    iterable = tqdm.tqdm(creatures, position=cpu_index, desc=f'Process {cpu_index+1}', dynamic_ncols=True, leave=False) if progressbar else creatures
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
