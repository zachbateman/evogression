'''
Module containing evolution algorithm for regression.
'''
import typing
import copy
import random
import tqdm
from pprint import pprint as pp
from .creatures import EvogressionCreature


class CreatureEvolution():

    def __init__(self,
                         target_parameter: str,
                         all_data: typing.List[typing.Dict[str, float]],
                         initial_num_creatures: int=50000) -> None:

        self.target_parameter = target_parameter
        self.all_data = all_data
        random.shuffle(self.all_data)
        self.training_data = self.all_data[:int(round(len(self.all_data) * 0.75))]
        self.testing_data = self.all_data[int(round(len(self.all_data) * 0.75)):]
        self.initial_num_creatures = initial_num_creatures

        self.creatures = [EvogressionCreature(target_parameter, full_parameter_example=self.all_data[0], layers=3, hunger=150 * random.random() + 10) for _ in range(initial_num_creatures)]
        self.current_generation = 1

        self.feast_num_food = 30
        self.famine_num_food = 10

        self.best_creatures = []
        self.evolve_creatures()


    def evolve_creatures(self):
        feast_or_famine = 'famine'
        counter = 1
        best_creature = None
        best_error = -1
        while True:
            print('-----------------------------------------')
            print(f'Cycle - {counter} -')
            print(f'Current Phase: {feast_or_famine}')
            self.evolution_cycle(feast_or_famine)

            # feast_or_famine = 'feast' if feast_or_famine == 'famine' else 'famine'
            if len(self.creatures) < 0.15 * self.initial_num_creatures:
                feast_or_famine = 'feast'
            elif len(self.creatures) > 0.15 * self.initial_num_creatures:
                feast_or_famine = 'famine'

            best_creature, error, avg_error = find_best_creature(self.creatures, self.target_parameter, self.training_data)

            self.best_creatures.append([copy.deepcopy(best_creature), error])
            print(f'Total number of creatures:  {len(self.creatures)}')
            print(f'Average Hunger: {self.average_creature_hunger}')
            print(f'Average error: ' + '{0:.2E}'.format(avg_error))
            print('Best Creature:')
            print(f'  Generation: {best_creature.generation}    error: ' + '{0:.2E}'.format(error))
            print()

            for creature_list in self.best_creatures:
                if creature_list[1] < best_error or best_error < 0:
                    best_error = creature_list[1]
                    best_creature = creature_list[0]

            # sprinkle in additional best_creatures to enhance this behaviour
            # also add in 3 of their offspring (mutated but close to latest best_creature)
            additional_best_creatures = [copy.deepcopy(best_creature) for _ in range(3)]
            additional_best_creatures.extend([additional_best_creatures[0] + additional_best_creatures[1] for _ in range(3)])
            additional_best_creatures = [cr for cr in additional_best_creatures if cr is not None]  # due to chance of not mating
            for cr in additional_best_creatures:
                cr.hunger = 100
            self.creatures.extend(additional_best_creatures)  # sprinkle in additional best_creatures to enhance this behaviour

            counter += 1
            if counter % 20 == 0:
                print('\n' * 3)
                print(f'BEST CREATURE AFTER {counter} ITERATIONS...')

                print(best_creature)
                print(f'Total Error: ' + '{0:.2E}'.format(best_error))

                if counter > 50:
                    self.best_creatures = self.best_creatures[20:]

                if counter > 1000:
                    breakpoint()



    def evolution_cycle(self, feast_or_famine: str):

        if feast_or_famine == 'feast':
            num_food = self.feast_num_food
        elif feast_or_famine == 'famine':
            num_food = self.famine_num_food

        random.shuffle(self.creatures)
        # "feed" groups of 5 creatures at a time.
        # creature with closest calc_target() to target gets to "eat" the data
        for i in range(0, len(self.creatures) // 5):
            creature_group = self.creatures[i: i + 5]
            for food_data in [random.choice(self.training_data) for _ in range(num_food)]:
                best_error = None
                best_creature = None
                error = 0
                for creature in creature_group:
                    target_calc = creature.calc_target(food_data)
                    error += abs(target_calc - food_data[self.target_parameter]) ** 2
                    if best_error is None or error < best_error:
                        best_error = error
                        best_creature = creature
                best_creature.hunger += 1

        self.run_metabolism_creatures()
        self.kill_weak_creatures()
        self.adjust_feast_famine_food_count(feast_or_famine)
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

    def adjust_feast_famine_food_count(self, feast_or_famine: str):
        '''
        Increase or decrease self.feast_num_food or self.famine_num_food
        so that population grows in feasting and shrinks in famine.
        '''
        if feast_or_famine == 'feast':
            if self.average_creature_hunger < 90:
                self.feast_num_food += 1
            if self.average_creature_hunger > 120:
                self.feast_num_food -= 1
            if len(self.creatures) < 0.1 * self.initial_num_creatures:
                self.feast_num_food += 10
            if self.feast_num_food <= self.famine_num_food:
                self.feast_num_food += 1
        elif feast_or_famine == 'famine':
            if self.average_creature_hunger < 50:
                self.famine_num_food += 1
            if self.average_creature_hunger > 80:
                self.famine_num_food -= 1
            if len(self.creatures) > 0.5 * self.initial_num_creatures:
                self.famine_num_food -= 3
            if self.famine_num_food >= self.feast_num_food:
                self.famine_num_food -= 1
            if self.famine_num_food < 1:
                self.famine_num_food = 1

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


def find_best_creature(creatures: list, target_parameter: str, data: list) -> tuple:

    best_error = -1  # to start loop
    avg_error = 0
    best_creature = None
    for creature in creatures:
        error = 0
        for data_point in data:
            target_calc = creature.calc_target(data_point)
            error += abs(target_calc - data_point[target_parameter]) ** 2
            avg_error += error
        if error < best_error or best_error < 1:
            best_error = error
            best_creature = creature
    avg_error /= len(creatures)
    return best_creature, best_error, avg_error

