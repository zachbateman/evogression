'''
Module containing "creatures" that each represent a potential regression equation.
'''
import random


class EvogressionCreature():


    def __init__(self,
                        target_parameter: str,
                        layers: int=1,
                        generation: int=1,
                        full_parameter_example: dict={},
                        modifiers: dict={}) -> None:

        # hunger decreases over time proportional to this creature's regression complexity
        # successfully "eating" a sample will increase self.hunger
        # creature dies when self.hunger == 0
        self.hunger = 100
        self.layers = layers
        self.generation = generation
        self.mutability = 5

        self.target_parameter = target_parameter

        # for any given input, want to modify/evolve as following:
        # inputs = {'a': 5.7, 'b': 3.2, 'c': 4.3}
        # target = 17.9
        # C_1 * (B_1 * a + Z_1) ** X_1 + C_2 * (B_2 * b + Z_2) ** X_2 + ...  + N_1 = T_1
        # T_1 = target (17.9) in single layer, else, feed T_1 as additional arg into next layer, etc.

        if modifiers != {}:
            if full_parameter_example == {}:
                print('Warning!  No modifiers or parameters provided to create EvogressionCreature!')
            self.modifiers = self.create_initial_modifiers(full_parameter_example)
        else:
            self.modifiers = modifiers



    def create_initial_modifiers(self, full_parameter_example) -> dict:
        # creates initial modifiers used with each given parameter
        modifiers = {}
        for layer in range(1, self.layers + 1):
            modifiers[f'LAYER_{layer}'] = {}
            for param in full_parameter_example.keys():
                # resist using parameters if many of them
                if random.random() < 1 / len(full_parameter_example) and param != self.target_parameter:
                    mut = self.mutability
                    C = 1 if random.random() < 0.8 else random.gauss(1, mut)
                    B = 1 if random.random() < 0.4 else random.gauss(1, 2 * mut)
                    Z = 0 if random.random() < 0.6 else random.gauss(0, 3 * mut)
                    X = 1 if random.random() < 0.9 else random.gauss(1, 0.1 * mut)
                    modifiers[f'LAYER_{layer}'][param] = {'C': C, 'B': B, 'Z': Z, 'X': X}
            if layer > 1:
                mut = self.mutability
                C = 1 if random.random() < 0.8 else random.gauss(1, mut)
                B = 1 if random.random() < 0.4 else random.gauss(1, 2 * mut)
                Z = 0 if random.random() < 0.6 else random.gauss(0, 3 * mut)
                X = 1 if random.random() < 0.9 else random.gauss(1, 0.1 * mut)
                modifiers[f'LAYER_{layer}']['T'] = {'C': C, 'B': B, 'Z': Z, 'X': X}
            modifiers[f'LAYER_{layer}']['N'] = 0 if random.random() < 0.2 else random.gauss(0, self.mutability)
        return modifiers

    def calc_target(self, parameters: dict) -> float:
        '''Apply the creature's modifiers to the parameters to calculate an attempt at target'''
        T = None  # has to be None on first layer
        for layer in range(1, self.layers + 1):
            T = self._calc_single_layer_target(parameters, layer, previous_T=T)
        return T

    def _calc_single_layer_target(self, parameters: dict, layer: int, previous_T=None) -> float:
        '''Apply creature's modifiers to parameters of ONE LAYER to calculate or help calculate target'''
        T = 0
        layer_name = f'LAYER_{layer}'
        for param, value in parameters.items():
            if param in self.modifiers[layer_name]:
                mods = self.modifiers[layer_name][param]
                T += mods['C'] * (mods['B'] * value + mods['Z']) ** mods['X']
        if previous_T and 'T' in self.modifiers[layer_name]:
            mods = self.modifiers[layer_name]['T']
            T += mods['C'] * (mods['B'] * previous_T + mods['Z']) ** mods['X']

        T += self.modifiers[layer_name]['N']
        return T

    @property
    def complexity_cost(self):
        '''
        Calculate an int to reduce hunger by based on the complexity of this creature.
        The idea is to penalize more complex models and thereby create a tendancy
        to develop a simpler model.
        '''
        cost = 0
        cost += 5 * self.layers
        for layer, layer_dict in self.modifiers.items():
            cost += len(layer_dict)
        return cost

    def __lt__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __add__(self, other):
        pass

    def __copy__(self):
        return EvogressionCreature(self.target_parameter, layers=self.layers, generation=self.generation, modifiers=self.modifiers)

    def __repr__(self) -> str:
        return 'EvogressionCreature'

    def get_regression_func(self):
        return self.modifiers
