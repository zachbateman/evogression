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
            self.creatures = easy_multip.map(generate_initial_creature, [arg_tup for _ in range(target_num_creatures)])
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
        acceptable_types = {'float', 'int', 'float64'}
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
        '''
        num_additional_best_creatures = int(round(0.005 * self.target_num_creatures, 0))
        return [self.best_creature.mutate_to_new_creature() for _ in range(num_additional_best_creatures)]


    def evolve_creatures(self, evolution_cycle_func=None, use_feast_and_famine=False):
        if evolution_cycle_func is None:
            evolution_cycle_func = self.evolution_cycle
        feast_or_famine = 'famine'
        counter = 1
        best_creature, best_error, new_best_creature = None, -1, False
        while True:
            print('-----------------------------------------')
            print(f'Cycle - {counter} -')
            if use_feast_and_famine:
                print(f'Current Phase: {feast_or_famine}')

            best_creature, error, median_error = self.calculate_all_and_find_best_creature()

            self.current_median_error = median_error
            self.best_creatures.append([copy.deepcopy(best_creature), error])
            self.print_cycle_stats(best_creature=best_creature, error=error, median_error=median_error, best_creature_error=error)

            for creature_list in self.best_creatures:
                if creature_list[1] < best_error or best_error < 0:
                    best_error = creature_list[1]
                    self.best_error = best_error
                    self.best_creature = creature_list[0]
                    new_best_creature = True
                    for param in self.best_creature.used_parameters():  # only count parameter usage for each NEW best_creature
                        self.parameter_usefulness_count[param] += 1

            self.creatures.extend(self.additional_best_creatures())  # sprinkle in additional best_creature mutants

            if counter == 1 or new_best_creature:
                pp(self.parameter_usefulness_count)
                print(f'\n\n\nNEW BEST CREATURE AFTER {counter} ITERATIONS...')
                print(self.best_creature)
                print('Total Error: ' + '{0:.2E}'.format(error))
                new_best_creature = False


            counter = self.check_cycles(counter)
            if self.num_cycles > 0 and counter == self.num_cycles:
                break

            if use_feast_and_famine:
                evolution_cycle_func(feast_or_famine)
                feast_or_famine = 'feast' if len(self.creatures) < self.target_num_creatures else 'famine'
            else:
                evolution_cycle_func()


    def evolution_cycle(self, feast_or_famine: str):
        '''Run one cycle of evolution'''
        # Option to add random new creatures each cycle (2.0% of target_num_creatures each time)
        if self.add_random_creatures_each_cycle:
            self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], hunger=80 * random.random() + 10, layers=self.force_num_layers) for _ in range(int(round(0.02 * self.target_num_creatures, 0)))])

        random.shuffle(self.creatures)  # used to mix up new creatures in among multip and randomize feeding groups
        self.run_metabolism_creatures()
        self.kill_weak_creatures()
        self.mate_creatures()


    def check_cycles(self, counter):
        '''Used in evolve_creatures() to handle cycle counting extras'''
        if counter % 10 == 0:
            if counter > 10:
                self.best_creatures = self.best_creatures[10:]
            if counter >= 10 and not self.num_cycles > 0:  # only pause if running indefinitely
                breakpoint()
        return counter + 1


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


    def return_best_creature(self):
        '''Return current best creature and standardizer if used'''
        error = -1
        for creature_list in self.best_creatures:
            if creature_list[1] < error or error < 0:
                error = creature_list[1]
                best_creature = creature_list[0]
        if self.standardize:
            return best_creature, self.standardizer
        else:
            return best_creature


    def stats_from_find_best_creature_multip_result(self, result_data) -> tuple:
        '''
        Unpack and return metrics from the data provided by the multip version of find_best_creature.
        '''
        calculated_creatures = []
        best_creature_lists = [result_data[5 * i: 5 * (i + 1)] for i in range(int(len(result_data) / 5))]
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

    def calculate_all_and_find_best_creature(self) -> tuple:
        if self.use_multip:
            if self.standardize:
                result_data = find_best_creature_multip(self.creatures, self.target_parameter, self.standardized_all_data, standardizer=self.standardizer, all_data_error_sums=self.all_data_error_sums)
            else:
                result_data = find_best_creature_multip(self.creatures, self.target_parameter, self.all_data, all_data_error_sums=self.all_data_error_sums)
            best_creature, error, median_error, calculated_creatures = self.stats_from_find_best_creature_multip_result(result_data)
        else:
            best_creature, error, median_error, calculated_creatures, all_data_error_sums = find_best_creature(self.creatures, self.target_parameter, self.standardized_all_data, all_data_error_sums=self.all_data_error_sums)
            self.all_data_error_sums = {**self.all_data_error_sums, **all_data_error_sums}
        self.creatures = calculated_creatures
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
        new_creatures_append = new_creatures.append
        for i in range(0, len(self.creatures), 2):
            creature_group = self.creatures[i: i + 2]
            try:
                new_creature = creature_group[0] + creature_group[1]
                if new_creature:
                    new_creatures_append(new_creature)
            except IndexError:
                pass
        self.creatures.extend(new_creatures)


    def optimize_best_creature(self, iterations=30):
        '''
        Use the creature.mutate_to_new_creature method to transform
        the best_creature into an even better fit.
        '''
        print('\n\n\nOptimizing best creature...')
        best_creature = self.best_creature
        pp(best_creature.modifiers)
        for _ in tqdm.tqdm(range(iterations)):
            mutated_clones = [best_creature] + [best_creature.mutate_to_new_creature() for _ in range(1000)]
            if self.use_multip:
                result_data = find_best_creature_multip(mutated_clones, self.target_parameter, self.standardized_all_data, all_data_error_sums=self.all_data_error_sums, progressbar=False)
                best_creature, error, median_error, calculated_creatures = self.stats_from_find_best_creature_multip_result(result_data)
            else:
                best_creature, error, median_error, calculated_creatures, all_data_error_sums = find_best_creature(mutated_clones, self.target_parameter, self.standardized_all_data, all_data_error_sums=self.all_data_error_sums, progressbar=False)
                self.all_data_error_sums = {**self.all_data_error_sums, **all_data_error_sums}
            print(f'Best error: ' + '{0:.6E}'.format(error))
        pp(best_creature.modifiers)
        self.best_creature = best_creature
        print('Best creature optimized!\n')


    def output_best_regression_function_as_module(self, output_filename='regression_function'):
        if self.standardize:
            self.best_creature.output_python_regression_module(output_filename=output_filename, standardizer=self.standardizer, directory='regression_modules', name_ext=f'___{round(self.best_error, 5)}')
        else:
            self.best_creature.output_python_regression_module(output_filename=output_filename, directory='regression_modules', name_ext=f'___{round(self.best_error, 5)}')




class CreatureEvolutionFittest(CreatureEvolution):
    '''
    Evolves creatures by killing off the worst performers in
    each cycle and then randomly generating many new creatures.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.evolve_creatures(self.evolution_cycle)
            if kwargs.get('optimize', True):
                self.optimize_best_creature()


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
        self.creatures = [creature for creature in self.creatures if error_sums.get(creature.modifier_hash, 0) > median_error]




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
    target_param, full_param_example, layers = arg_tup
    return EvogressionCreature(target_param, full_parameter_example=full_param_example, hunger=80 * random.random() + 10, layers=layers)


def feed_creature_groups(arg_tup):
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
