'''
Module containing evolution algorithm for regression.
'''
import typing
import copy
import random
import tqdm
import warnings
import easy_multip
from pprint import pprint as pp
from .creatures import EvogressionCreature
from .standardize import Standardizer



class CreatureEvolution():

    feast_group_size = 2
    famine_group_size = 50

    def __init__(self,
                       target_parameter: str,
                       all_data: typing.List[typing.Dict[str, float]],
                       target_num_creatures: int=30000,
                       add_random_creatures_each_cycle: bool=True,
                       num_cycles: int=0,
                       force_num_layers: int=0,
                       standardize: bool=True,
                       use_multip: bool=True,
                       initial_creature_creation_multip: bool=True) -> None:

        self.target_parameter = target_parameter
        self.standardize = standardize
        self.all_data = all_data
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
            self.creatures = easy_multip.map(generate_initial_creature, [arg_tup for _ in range(int(round(1.1 * target_num_creatures, 0)))])
        else:
            self.creatures = [generate_initial_creature(arg_tup) for _ in range(int(round(1.1 * target_num_creatures, 0)))]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.evolve_creatures()



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


    def evolve_creatures(self):
        feast_or_famine = 'famine'
        counter = 1
        best_creature = None
        best_error = -1
        new_best_creature = False
        while True:
            print('-----------------------------------------')
            print(f'Cycle - {counter} -')
            print(f'Current Phase: {feast_or_famine}')

            if self.use_multip:
                calculated_creatures = []
                if self.standardize:
                    result_data = find_best_creature_multip(self.creatures, self.target_parameter, self.standardized_all_data, standardizer=self.standardizer, all_data_error_sums=self.all_data_error_sums)
                else:
                    result_data = find_best_creature_multip(self.creatures, self.target_parameter, self.all_data, all_data_error_sums=self.all_data_error_sums)
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
            else:
                best_creature, error, median_error, calculated_creatures, all_data_error_sums = find_best_creature(self.creatures, self.target_parameter, self.standardized_all_data, all_data_error_sums=self.all_data_error_sums)
                self.all_data_error_sums = {**self.all_data_error_sums, **all_data_error_sums}
            self.creatures = calculated_creatures

            for param in best_creature.used_parameters():
                self.parameter_usefulness_count[param] += 1

            self.best_creatures.append([copy.deepcopy(best_creature), error])
            print(f'Total number of creatures:  {len(self.creatures)}')
            print(f'Average Hunger: {round(self.average_creature_hunger, 1)}')
            print(f'Median error: ' + '{0:.2E}'.format(median_error))
            print('Best Creature:')
            print(f'  Generation: {best_creature.generation}    Error: ' + '{0:.2E}'.format(error))

            for creature_list in self.best_creatures:
                if creature_list[1] < best_error or best_error < 0:
                    best_error = creature_list[1]
                    self.best_creature = creature_list[0]
                    new_best_creature = True


            self.creatures.extend(self.additional_best_creatures())  # sprinkle in additional best_creatures to enhance the top-performing behaviour

            if counter == 1 or new_best_creature:  # self.best_creatures[-1][0].modifiers != self.best_creatures[-2][0].modifiers:
                print(f'\n\n\nNEW BEST CREATURE AFTER {counter} ITERATIONS...')
                print(self.best_creature)
                print(f'Total Error: ' + '{0:.2E}'.format(error))
                new_best_creature = False

                pp(self.parameter_usefulness_count)

            if self.num_cycles > 0 and counter == self.num_cycles:
                break

            if counter % 10 == 0:
                if counter > 10:
                    self.best_creatures = self.best_creatures[10:]
                if counter >= 10 and not self.num_cycles > 0:  # only pause if running indefinitely
                    breakpoint()

            self.evolution_cycle(feast_or_famine)

            feast_or_famine = 'feast' if len(self.creatures) < self.target_num_creatures else 'famine'
            counter += 1


    def additional_best_creatures(self) -> list:
        '''
        sprinkle in additional best_creatures to enhance this behavior
        also add in their offspring (mutated but close to latest best_creature)
        '''
        num_additional_best_creatures = int(round(0.005 * self.target_num_creatures, 0))
        additional_best_creatures = [copy.deepcopy(self.best_creature) for _ in range(num_additional_best_creatures)]
        additional_best_creatures.extend([additional_best_creatures[0] + additional_best_creatures[1] for _ in range(num_additional_best_creatures)])
        additional_best_creatures = [cr for cr in additional_best_creatures if cr is not None]  # due to chance of not mating
        return additional_best_creatures


    def evolution_cycle(self, feast_or_famine: str):

        # Option to add random new creatures each cycle (2.0% of target_num_creatures each time)
        if self.add_random_creatures_each_cycle:
            self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], hunger=80 * random.random() + 10, layers=self.force_num_layers) for _ in range(int(round(0.02 * self.target_num_creatures, 0)))])

        random.shuffle(self.creatures)  # used to mix up new creatures in among multip and randomize feeding groups
        self.feed_creatures(feast_or_famine)
        self.run_metabolism_creatures()
        self.kill_weak_creatures()
        self.mate_creatures()


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

        random_choice = random.choice  # local variable for speed
        if self.use_multip:
            creature_groups = (creature_group for creature_group in (self.creatures[group_size * i:group_size * (i + 1)] for i in range(0, len(self.creatures) // group_size)))
            food_groups = ([random_choice(all_food_data) for _ in range(5)] for i in range(0, len(self.creatures) // group_size))
            feeding_arg_tups = [(creature_group, food_group, self.target_parameter, self.standardizer) for creature_group, food_group in zip(creature_groups, food_groups)]
            hunger_modified_creatures = easy_multip.map(feed_creature_groups, feeding_arg_tups)
            self.creatures = [creature for sublist in hunger_modified_creatures for creature in sublist]
        else:
            target_parameter = self.target_parameter  # local variable for speed
            standardizer = self.standardizer  # local variable for speed
            for i in range(0, len(self.creatures) // group_size):
                creature_group = self.creatures[group_size * i:group_size * (i + 1)]
                for food_data in [random_choice(all_food_data) for _ in range(5)]:
                    best_error, best_creature = None, None
                    for creature in creature_group:
                        error = calc_error_value(creature, target_parameter, food_data, standardizer)
                        if best_error is None or error < best_error:
                            best_error, best_creature = error, creature
                    best_creature.hunger += 6


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


    def mate_creatures(self):
        '''
        Mate creatures to generate new creatures.
        '''
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


    def output_best_regression_function_as_module(self, output_filename='regression_function.py'):
        if self.standardize:
            self.best_creature.output_python_regression_module(output_filename=output_filename, standardizer=self.standardizer)
        else:
            self.best_creature.output_python_regression_module(output_filename=output_filename)


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
        error = abs(target_calc - data_point_calc) ** 2.0  # sometimes generates "RuntimeWarning: overflow encountered in double_scalars"
    except OverflowError:  # if error is too big to store, give huge arbitrary error
        error = 10 ** 150
    return error


def find_best_creature(creatures: list, target_parameter: str, data: list, standardizer=None, all_data_error_sums: dict={}) -> tuple:
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
    for creature in tqdm.tqdm(creatures):
        try:
            error = all_data_error_sums[creature.modifier_hash]
        except KeyError:
            error = sum([calc_error_value(creature, target_parameter, data_point, standardizer) for data_point in data])
            all_data_error_sums[creature.modifier_hash] = error

        append_to_calculated_creatures(creature)
        error /= data_length
        if error < best_error or best_error < 0:
            best_error = error
            best_creature = creature
        append_to_errors(error)
    avg_error = sorted(errors)[len(errors) // 2]
    return [best_creature, best_error, avg_error, calculated_creatures, all_data_error_sums]
find_best_creature_multip = easy_multip.decorators.use_multip(find_best_creature)
