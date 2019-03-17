'''
Module containing evolution algorithm for regression.
'''
import typing
import copy
import random
from .creatures import EvogressionCreature


# all_data = [{}, {}, ...]  # actual data provided in list of dicts
# initial_creatures = [EvogressionCreature({'x': None, 'y': None}, 'y', layers=3) for _ in range(10000)]


class CreatureEvolution():

    def __init__(self,
                         target_parameter: str,
                         all_data: typing.List[typing.Dict[str, float]],
                         initial_num_creatures: int=10000) -> None:

        self.all_data = random.shuffle(all_data)
        self.training_data = self.all_data[:int(round(len(self.all_data) * 0.75))]
        self.testing_data = self.all_data[int(round(len(self.all_data) * 0.75)):]
        self.initial_num_creatures = initial_num_creatures

        self.creatures = [EvogressionCreature(target_parameter, {'x': None, 'y': None}, 'y', layers=3) for _ in range(initial_num_creatures)]
        self.current_generation = 1

        self.feast_num_food = 15
        self.famine_num_food = 5



        def feast_evolution_cycle(self, feast_or_famine: str):

            if feast_or_famine == 'feast':
                num_food = self.feast_num_food
            elif feast_or_famine == 'famine':
                num_food = self.famine_num_food

            random.shuffle(self.creatures)
            # "feed" groups of 3 creatures at a time.
            # creature with closest calc_target() to target gets to "eat" the data
            for i in range(0, len(self.creatures) / 3):
                creature_group = self.creatures[i: i + 3]
                for food_data in [random.choice(self.training_data) for _ in range(num_food)]:
                    best_error = None
                    best_creature = None
                    for creature in creature_group:
                        target_calc = creature.calc_target(food_data)
                        error += abs(target_calc - linear_data['y'][index]) ** 2
                        if best_error is None or error < best_error:
                            best_error = error
                            best_creature = creature
                    best_creature.hunger += 1

            self.run_metabolism_creatures()
            self.kill_0_hunger_creatures()
            self.adjust_feast_famine_food_count(feast_or_famine)
            self.mate_creatures()


        def run_metabolism_creatures(self):
            '''Deduct from each creature's hunger as their complexity demands'''
            for creature in self.creatures:
                creature.hunger -= creature.complexity_cost

        def kill_0_hunger_creatures(self):
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
                if self.average_creature_hunger < 100:
                    self.feast_num_food += 1
                if self.average_creature_hunger > 150:
                    self.feast_num_food -= 1
                if self.feast_num_food <= self.famine_num_food:
                    self.feast_num_food += 1
            elif feast_or_famine == 'famine':
                if self.average_creature_hunger < 20:
                    self.famine_num_food += 1
                if self.average_creature_hunger > 80:
                    self.famine_num_food -= 1
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
                creature_group = self.creatures[i: i + 1]
                new_creature = creature_group[0] + creature_group[1]
                if new_creature:
                    new_creatures.append(new_creature)
            self.creatures.extend(new_creatures)
