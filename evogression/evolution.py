'''
Module containing evolution algorithms for regression.
'''
from typing import List, Dict
import statistics
import copy
import random
import tqdm
import warnings
import easy_multip
from pprint import pprint as pp
from .creatures import EvogressionCreature
from .standardize import Standardizer



class CreatureEvolution():
    '''
    Creates a framework for evolving groups of creatures.
    This class is designed to be subclassed into more
    specific evolution algorithms.
    '''
    feast_group_size = 2
    famine_group_size = 50

    def __init__(self,
                 target_parameter: str,
                 all_data: List[Dict[str, float]],
                 target_num_creatures: int=30000,
                 add_random_creatures_each_cycle: bool=True,
                 num_cycles: int=0,
                 force_num_layers: int=0,
                 standardize: bool=True,
                 use_multip: bool=True,
                 fill_none: bool=True,
                 initial_creature_creation_multip: bool=True,
                 **kwargs) -> None:

        self.target_parameter = target_parameter
        self.standardize = standardize
        self.all_data = all_data
        if fill_none:
            self.fill_none_with_median()

        self.target_num_creatures = target_num_creatures
        self.add_random_creatures_each_cycle = add_random_creatures_each_cycle
        self.num_cycles = num_cycles
        self.force_num_layers = force_num_layers
        self.use_multip = use_multip

        self.current_generation = 1
        self.all_data_error_sums: dict = {}
        self.best_creatures: list = []
        self.parameter_usefulness_count: dict={key: 0 for key in all_data[0] if key != target_parameter}

        if self.standardize:
            self.standardizer = Standardizer(self.all_data)
            self.standardized_all_data = self.standardizer.get_standardized_data()
        else:
            self.standardizer = None

        self.data_checks()

        arg_tup = (target_parameter, self.all_data[0], force_num_layers)
        if initial_creature_creation_multip:
            try:
                self.creatures = easy_multip.map(generate_initial_creature, [arg_tup for _ in range(target_num_creatures)])
            except RuntimeError:  # multiprocessing RuntimeError occurs if code not within if __name__ block
                raise RuntimeError('\n  When using multiprocessing, please ensure your code is running from within a\n  if __name__ == \'__main__\': block!')
        else:
            self.creatures = [generate_initial_creature(arg_tup) for _ in range(target_num_creatures)]


    def fill_none_with_median(self):
        '''
        Find median value of each input parameter and
        then replace any None values with this median.
        '''
        parameters_adjusted = []
        for param in self.all_data[0].keys():
            if param != self.target_parameter:
                values = [d[param] for d in self.all_data if d[param] is not None]
                median = statistics.median(values)
                for d in self.all_data:
                    if d[param] is None:
                        parameters_adjusted.append(param)
                        d[param] = median
        # Remove any data points that have None for the target/result parameter
        self.all_data = [d for d in self.all_data if d[self.target_parameter] is not None]

        if len(parameters_adjusted) >= 1:
            print('Data None values filled with median for the following parameters:')
            for param in sorted(set(parameters_adjusted)):
                print(f'  {param}')


    def data_checks(self):
        '''Check input data for potential issues'''
        acceptable_types = {'float', 'int', 'float64', 'int64'}
        def check_data(data, data_name):
            for i, d in enumerate(data):
                for key, val in d.items():
                    # val > & < checks are way of checking for nan without needing to require numpy import
                    if type(val).__name__ not in acceptable_types or not (val >= 0 or val <= 0):
                        print(f'ERROR!  NAN values detected in {data_name}!')
                        print(f'Index: {i}  key: {key}  value: {val}  type: {type(val).__name__}')
                        breakpoint()
        check_data(self.all_data, 'all_data')
        check_data(self.standardized_all_data, 'standardized_all_data')


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


    def evolve_creatures(self, evolution_cycle_func=None, use_feast_and_famine=False, progressbar=True):
        '''
        Main evolution loop that handles results of each loop and
        keeps track of best creatures/regression equations.
        '''
        if evolution_cycle_func is None:
            evolution_cycle_func = self.evolution_cycle

        counter = 0
        while True:
            counter += 1
            print('-----------------------------------------')
            print(f'Cycle - {counter} -')
            if use_feast_and_famine:
                print(f'Current Phase: {feast_or_famine}')

            best_creature, error, median_error = self.calculate_all_and_find_best_creature(progressbar=progressbar)

            self.current_median_error = median_error
            new_best_creature = self.record_best_creature(best_creature, error)
            self.print_cycle_stats(best_creature=best_creature, error=error, median_error=median_error, best_creature_error=error)

            if new_best_creature:
                print(f'\n\n\nNEW BEST CREATURE AFTER {counter} ITERATION{"S" if counter > 1 else ""}...')
                print(best_creature)
                print('Total Error: ' + '{0:.2E}'.format(error))

            self.creatures.extend(self.additional_best_creatures())  # sprinkle in additional best_creature mutants

            if counter >= self.num_cycles:
                break

            if use_feast_and_famine:
                feast_or_famine = 'famine' if counter <= 2 else feast_or_famine
                evolution_cycle_func(feast_or_famine)
                feast_or_famine = 'feast' if len(self.creatures) < self.target_num_creatures else 'famine'
            else:
                evolution_cycle_func()


    def evolution_cycle(self, feast_or_famine: str):
        '''
        Default evolution cycle.
        Run one cycle of evolution that introduces new random creatures,
        kills weak creatures, and mates the remaining ones.
        '''
        # Option to add random new creatures each cycle (2.0% of target_num_creatures each time)
        if self.add_random_creatures_each_cycle:
            self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], hunger=80 * random.random() + 10, layers=self.force_num_layers) for _ in range(int(round(0.02 * self.target_num_creatures, 0)))])

        random.shuffle(self.creatures)  # used to mix up new creatures in among multip and randomize feeding groups
        self.run_metabolism_creatures()
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


    def shrink_error_cache(self):
        '''
        Delete first portion of cache to keep from growing forever
        Python dictionaries are now insertion-ordered, so this should work well
        '''
        hash_keys = list(self.all_data_error_sums.keys())
        if len(hash_keys) > self.target_num_creatures * 3:
            for key in hash_keys[:self.target_num_creatures]:
                del self.all_data_error_sums[key]


    def run_metabolism_creatures(self):
        '''Deduct from each creature's hunger as their complexity demands'''
        for creature in self.creatures:
            creature.hunger -= creature.complexity_cost


    def kill_weak_creatures(self):
        '''Remove all creatures whose hunger has dropped to 0 or below'''
        self.creatures = [creature for creature in self.creatures if creature.hunger > 0]


    @property
    def average_creature_hunger(self):
        return sum(c.hunger for c in self.creatures) / len(self.creatures)


    def return_best_creature(self, with_error=False):
        '''Return current best creature and standardizer if used'''
        print('\nDEPRECATION WARNING: "evolution.return_best_creature"')
        print('  Use evolution.best_creature, evolution.best_error, or evolution.standardizer instead!\n')
        error = -1
        for creature_list in self.best_creatures:
            if creature_list[1] < error or error < 0:
                error = creature_list[1]
                best_creature = creature_list[0]
        if self.standardize:
            if with_error:
                return best_creature, error, self.standardizer
            else:
                return best_creature, self.standardizer
        else:
            if with_error:
                return best_creature, error
            else:
                return best_creature


    def stats_from_find_best_creature_multip_result(self, result_data: list) -> tuple:
        '''
        Unpack and return metrics from the data provided by the multip version of find_best_creature.

        result_data arg is a list where each 5 items is the output from one find_best_creature call.
        Specifically, result_data looks like:
            [EvogressionCreature, best_error, avg_error, calculated_creatures, all_data_error_sums,
              EvogressionCreature, ...,
              ...
            ]
        '''
        calculated_creatures = []
        best_creature_lists = [result_data[5 * i: 5 * (i + 1)] for i in range(int(len(result_data) / 5))]  # make result_data useful
        # best_creature_lists is list with items of form [best_creature, error, avg_error]
        error, best_creature = None, None
        for index, bc_list in enumerate(best_creature_lists):
            calculated_creatures.extend(bc_list[3])
            if error is None or bc_list[1] < error:
                error = bc_list[1]
                best_creature = bc_list[0]
            self.all_data_error_sums = {**self.all_data_error_sums, **bc_list[4]}  # recreate all_data_error_sums cache with results including updated cache values
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
                                                                              all_data_error_sums=self.all_data_error_sums, progressbar=progressbar)
            else:
                result_data = find_best_creature_multip(self.creatures, self.target_parameter,
                                                                              self.all_data, all_data_error_sums=self.all_data_error_sums,
                                                                              progressbar=progressbar)
            best_creature, error, median_error, calculated_creatures = self.stats_from_find_best_creature_multip_result(result_data)
        else:
            best_creature, error, median_error, calculated_creatures, all_data_error_sums = find_best_creature(self.creatures, self.target_parameter, self.standardized_all_data, all_data_error_sums=self.all_data_error_sums, progressbar=progressbar)
            self.all_data_error_sums = {**self.all_data_error_sums, **all_data_error_sums}
        self.creatures = calculated_creatures
        self.shrink_error_cache()
        return best_creature, error, median_error


    def print_cycle_stats(self, best_creature=None, error=None, median_error: float=0, best_creature_error: float=0) -> None:
        print(f'Total number of creatures:  {len(self.creatures)}')
        print(f'Average Hunger: {round(self.average_creature_hunger, 1)}')
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
        for i in tqdm.tqdm(range(iterations)):
            if i < iterations / 3:  # quickly mutate creature for first 1/3rd of iterations and then make small, fine mutations
                adjustments = 'fast'
            else:
                adjustments = 'fine'

            mutated_clones = [best_creature] + [best_creature.mutate_to_new_creature(adjustments=adjustments) for _ in range(500)]
            if self.use_multip:
                result_data = find_best_creature_multip(mutated_clones, self.target_parameter, self.standardized_all_data, all_data_error_sums=self.all_data_error_sums, progressbar=False)
                best_creature, error, median_error, calculated_creatures = self.stats_from_find_best_creature_multip_result(result_data)
            else:
                best_creature, error, median_error, calculated_creatures, all_data_error_sums = find_best_creature(mutated_clones, self.target_parameter, self.standardized_all_data, all_data_error_sums=self.all_data_error_sums, progressbar=False)
                self.all_data_error_sums = {**self.all_data_error_sums, **all_data_error_sums}
            print(f'Best error: ' + '{0:.6E}'.format(error))
            self.shrink_error_cache()
            errors.append(error)
            if i > iterations / 3 + 3 and iterations > 10 and error / errors[-3] > 0.999:
                break  # break out of the loop if it's no longer improving accuracy
        new_best_creature = self.record_best_creature(best_creature, error)
        print(self.best_creature)
        print('Best creature optimized!\n')


    def output_best_regression_function_as_module(self, output_filename='regression_function', add_error_value=True):
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


    def add_predictions_to_data(self, data: List[dict], standardized_data: bool=False) -> List[dict]:
        '''
        Add best_creature predictions to data arg as f'{target}_PREDICTED' new key.
        Return unstandardized list of dicts.
        '''
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




class CreatureEvolutionFittest(CreatureEvolution):
    '''
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

            self.all_data_error_sums = {}  # clear out all_data_error_sums dict to save memory


    def evolution_cycle(self):
        '''Run one cycle of evolution'''
        self.kill_weak_creatures()
        self.mate_creatures()

        # Add random new creatures each cycle to get back to target num creatures
        self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], hunger=80 * random.random() + 10, layers=self.force_num_layers) for _ in range(int(round(self.target_num_creatures - len(self.creatures), 0)))])
        random.shuffle(self.creatures)  # used to mix up new creatures in among multip


    def kill_weak_creatures(self):
        '''Overwrite CreatureEvolution's kill_weak_creatures method'''
        error_sums = self.all_data_error_sums
        median_error = self.current_median_error
        # next line uses .get(..., 0) to keep creatures if they aren't yet calculated (mutated best_creatures)
        self.creatures = [creature for creature in self.creatures if error_sums.get(creature.modifier_hash, 0) < median_error]




class CreatureEvolutionNatural(CreatureEvolution):
    '''
    Evolves creatures by "feeding" them.  The better creatures
    successfully model test data and stay healthy while bad
    performers get progressively "hungrier" until they are killed off.

    Cycles of "feast" and "famine" cause the community of creatures to
    grow and shrink with each phase either increasing the diversity of
    creatures (regression equations) or decreasing the diversity by
    killing off the lower-performing creatures.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.evolve_creatures(evolution_cycle_func=self.evolution_cycle, use_feast_and_famine=True)

            if kwargs.get('clear_creatures', False):  # save tons of memory when returning object (helps with multip)
                self.creatures = [self.best_creature]

            self.all_data_error_sums = {}  # clear out all_data_error_sums dict to save memory


    def feed_creatures(self, feast_or_famine: str):
        '''
        "feed" groups of creatures at a once.
        creature with closest calc_target() to target gets to "eat" the data
        '''
        if feast_or_famine == 'feast':
            group_size = self.feast_group_size
        elif feast_or_famine == 'famine':
            group_size = self.famine_group_size

        all_food_data = self.standardized_all_data if self.standardize else self.all_data

        rand_choice = random.choice  # local variable for speed
        if self.use_multip:
            creature_groups = (creature_group for creature_group in (self.creatures[group_size * i:group_size * (i + 1)] for i in range(0, len(self.creatures) // group_size)))
            food_groups = ([rand_choice(all_food_data) for _ in range(5)] for i in range(0, len(self.creatures) // group_size))
            feeding_arg_tups = [(creature_group, food_group, self.target_parameter, self.standardizer) for creature_group, food_group in zip(creature_groups, food_groups)]
            hunger_modified_creatures = easy_multip.map(feed_creature_groups, feeding_arg_tups)
            self.creatures = [creature for sublist in hunger_modified_creatures for creature in sublist]
        else:
            target_parameter = self.target_parameter  # local variable for speed
            standardizer = self.standardizer  # local variable for speed
            for i in range(0, len(self.creatures) // group_size):
                creature_group = self.creatures[group_size * i:group_size * (i + 1)]
                for food_data in [rand_choice(all_food_data) for _ in range(5)]:
                    best_error, best_creature = None, None
                    for creature in creature_group:
                        error = calc_error_value(creature, target_parameter, food_data, standardizer)
                        if best_error is None or error < best_error:
                            best_error, best_creature = error, creature
                    best_creature.hunger += 6


    def evolution_cycle(self, feast_or_famine: str):
        '''Run one cycle of evolution'''
        # Option to add random new creatures each cycle (2.0% of target_num_creatures each time)
        if self.add_random_creatures_each_cycle:
            self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], hunger=80 * random.random() + 10, layers=self.force_num_layers) for _ in range(int(round(0.02 * self.target_num_creatures, 0)))])

        random.shuffle(self.creatures)  # used to mix up new creatures in among multip and randomize feeding groups
        self.feed_creatures(feast_or_famine)
        self.run_metabolism_creatures()
        self.kill_weak_creatures()
        self.mate_creatures()




def generate_initial_creature(arg_tup):
    '''
    Creates an EvogressionCreature.
    This is a module-level function so that it can be used by multip.
    '''
    target_param, full_param_example, layers = arg_tup
    return EvogressionCreature(target_param, full_parameter_example=full_param_example, hunger=80 * random.random() + 10, layers=layers)


def feed_creature_groups(arg_tup):
    '''
    Calculate the error for each creature for a group of data points and
    increase the "hunger" for the creatures that are more accurate.
    '''
    creature_group, food_group, target_parameter, standardizer = arg_tup
    for food_data in food_group:
        best_error, best_creature = 10 ** 150, None
        for creature in creature_group:
            error = calc_error_value(creature, target_parameter, food_data, standardizer)
            if error < best_error:
                best_error, best_creature = error, creature
        best_creature.hunger += 6
    return creature_group


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


def find_best_creature(creatures: list, target_parameter: str, data: list, standardizer=None, all_data_error_sums: dict={}, progressbar=True) -> tuple:
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
        try:
            error = all_data_error_sums[creature.modifier_hash]
        except KeyError:
            error = sum([calc_error_value(creature, target_parameter, data_point, standardizer) for data_point in data]) / data_length
            all_data_error_sums[creature.modifier_hash] = error

        append_to_calculated_creatures(creature)
        if error < best_error or best_error < 0:
            best_error = error
            best_creature = creature
        append_to_errors(error)
    avg_error = sorted(errors)[len(errors) // 2]  # MEDIAN error
    return (best_creature, best_error, avg_error, calculated_creatures, all_data_error_sums)
find_best_creature_multip = easy_multip.decorators.use_multip(find_best_creature)
