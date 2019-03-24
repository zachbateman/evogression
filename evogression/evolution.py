'''
Module containing evolution algorithm for regression.
'''
import typing
import copy
import random
import tqdm
import numpy
from pprint import pprint as pp
import easy_multip
from .creatures import EvogressionCreature
from .standardize import Standardizer


class CreatureEvolution():

    def __init__(self,
                         target_parameter: str,
                         all_data: typing.List[typing.Dict[str, float]],
                         target_num_creatures: int=30000,
                         add_random_creatures_each_cycle: bool=True,
                         num_cycles: int=0,
                         force_num_layers: int=0,
                         standardize: bool=True) -> None:

        self.target_parameter = target_parameter
        self.standardize = standardize
        self.all_data = all_data
        random.shuffle(self.all_data)
        for i, d in enumerate(self.all_data):
            for key, val in d.items():
                if numpy.isnan(val):
                    print('ERROR!  NAN values detected in all_data!')
                    print(f'Index: {i}  data: {d}')
        self.training_data = self.all_data[:int(round(len(self.all_data) * 0.75))]
        self.testing_data = self.all_data[int(round(len(self.all_data) * 0.75)):]

        if self.standardize:
            self.standardizer = Standardizer(self.training_data)  # only use training data for Standardizer!! (must be blind to consider test data)
            self.standardized_training_data = self.standardizer.get_standardized_data()
        else:
            self.standardizer = None

        self.target_num_creatures = target_num_creatures
        self.add_random_creatures_each_cycle = add_random_creatures_each_cycle
        self.num_cycles = num_cycles
        self.force_num_layers = force_num_layers

        self.creatures = [EvogressionCreature(target_parameter, full_parameter_example=self.all_data[0], hunger=80 * random.random() + 10, layers=self.force_num_layers) for _ in range(int(round(1.1 * target_num_creatures, 0)))]
        self.current_generation = 1

        self.feast_group_size = 2
        self.famine_group_size = 50

        self.best_creatures = []
        self.evolve_creatures()


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

            if self.standardize:
                result_data =  find_best_creature(self.creatures, self.target_parameter, self.standardized_training_data, standardizer=self.standardizer)
            else:
                result_data =  find_best_creature(self.creatures, self.target_parameter, self.training_data)
            best_creature_lists = [result_data[3 * i: 3 * (i + 1)] for i in range(int(len(result_data) / 3))]
            # best_creature_lists is list with items of form [best_creature, error, avg_error]
            error = -1
            best_creature = None
            for index, bc_list in enumerate(best_creature_lists):
                if bc_list[1] < error or error < 0:
                    error = bc_list[1]
                    best_creature = bc_list[0]
            median_error = sum(bc_list[2] for bc_list in best_creature_lists) / len(best_creature_lists)  # mean of medians of big chunks...

            # best_creature, error, median_error = find_best_creature(self.creatures, self.target_parameter, self.standardized_training_data)

            self.best_creatures.append([copy.deepcopy(best_creature), error])
            print(f'Total number of creatures:  {len(self.creatures)}')
            print(f'Average Hunger: {round(self.average_creature_hunger, 1)}')
            print(f'Median error: ' + '{0:.2E}'.format(median_error))
            print('Best Creature:')
            print(f'  Generation: {best_creature.generation}    Error: ' + '{0:.2E}'.format(error))

            bc_error = sum(calc_error_value(best_creature, self.target_parameter, data_point, self.standardizer) for data_point in self.testing_data) / len(self.testing_data)
            print('  Testing Data Error:     ' + '{0:.2E}'.format(bc_error))
            print()

            for creature_list in self.best_creatures:
                if creature_list[1] < best_error or best_error < 0:
                    best_error = creature_list[1]
                    best_creature = creature_list[0]
                    new_best_creature = True

            # sprinkle in additional best_creatures to enhance this behavior
            # also add in 3 of their offspring (mutated but close to latest best_creature)
            additional_best_creatures = [copy.deepcopy(best_creature) for _ in range(int(round(0.005 * self.target_num_creatures, 0)))]
            additional_best_creatures.extend([additional_best_creatures[0] + additional_best_creatures[1] for _ in range(int(round(0.005 * self.target_num_creatures, 0)))])
            additional_best_creatures = [cr for cr in additional_best_creatures if cr is not None]  # due to chance of not mating
            for cr in additional_best_creatures:
                cr.hunger = 100
            self.creatures.extend(additional_best_creatures)  # sprinkle in additional best_creatures to enhance this behaviour

            if counter == 1 or new_best_creature: #self.best_creatures[-1][0].modifiers != self.best_creatures[-2][0].modifiers:
                print('\n' * 3)
                print(f'NEW BEST CREATURE AFTER {counter} ITERATIONS...')
                print(best_creature)
                print(f'Total Error: ' + '{0:.2E}'.format(error))
                new_best_creature = False

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


    def evolution_cycle(self, feast_or_famine: str):

        if feast_or_famine == 'feast':
            group_size = self.feast_group_size
        elif feast_or_famine == 'famine':
            group_size = self.famine_group_size

        random.shuffle(self.creatures)
        # "feed" groups of creatures at a once.
        # creature with closest calc_target() to target gets to "eat" the data
        all_food_data = self.training_data  # self.standardized_training_data if self.standardize else self.training_data
        for i in range(0, len(self.creatures) // group_size):
            creature_group = self.creatures[group_size * i:group_size * (i + 1)]
            for food_data in [random.choice(all_food_data) for _ in range(30)]:

                best_error = None
                best_creature = None
                error = 0
                for creature in creature_group:
                    error += calc_error_value(creature, self.target_parameter, food_data, self.standardizer)
                    if best_error is None or error < best_error:
                        best_error = error
                        best_creature = creature
                best_creature.hunger += 1

        self.run_metabolism_creatures()
        self.kill_weak_creatures()

        # Option to add random new creatures each cycle (2.0% of target_num_creatures each time)
        if self.add_random_creatures_each_cycle:
            self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], hunger=80 * random.random() + 10, layers=self.force_num_layers) for _ in range(int(round(0.02 * self.target_num_creatures, 0)))])

        self.mate_creatures()


    def run_metabolism_creatures(self):
        '''Deduct from each creature's hunger as their complexity demands'''
        for creature in self.creatures:
            creature.hunger -= creature.complexity_cost

    def kill_weak_creatures(self):
        '''Remove all creatures whose hunger has dropped to 0 or below'''
        for index, creature in enumerate(self.creatures):
            if creature.hunger <= 0:
                del self.creatures[index]

    @property
    def average_creature_hunger(self):
        return sum(c.hunger for c in self.creatures) / len(self.creatures)

    def return_best_creature(self):
        '''Return current best creature and standardizer'''
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
        for i in range(0, len(self.creatures), 2):
            creature_group = self.creatures[i: i + 2]
            try:
                new_creature = creature_group[0] + creature_group[1]
                if new_creature:
                    new_creatures.append(new_creature)
            except IndexError:
                pass
        self.creatures.extend(new_creatures)


def calc_error_value(creature, target_parameter: str, data_point: dict, standardizer=None) -> float:
    '''Calculate the error between a creature's predicted value and the actual value'''
    target_calc = creature.calc_target(data_point)
    data_point_calc = data_point[target_parameter]
    if standardizer is not None:
        target_calc = standardizer.unstandardize_value(target_parameter, target_calc)
        data_point_calc = standardizer.unstandardize_value(target_parameter, data_point_calc)
    try:
        error = abs(target_calc - data_point_calc) ** 2.0
    except OverflowError:  # if error is too big to store, give huge arbitrary error
        error = 10 ** 150
    return error



def _find_best_creature(creatures: list, target_parameter: str, data: list, standardizer=None) -> tuple:
    best_error = -1  # to start loop
    errors = []
    best_creature = None
    for creature in tqdm.tqdm(creatures):
        error = 0
        for data_point in data:
            error += calc_error_value(creature, target_parameter, data_point, standardizer)
        error /= len(data)
        if error < best_error or best_error < 1:
            best_error = error
            best_creature = creature
        errors.append(error)
    avg_error = sorted(errors)[len(errors)// 2]
    return [best_creature, best_error, avg_error]
find_best_creature = easy_multip.decorators.use_multip(_find_best_creature)
