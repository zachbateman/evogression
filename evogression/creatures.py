'''
Module containing "creatures" that each represent a potential regression equation.
'''
import random
from ._version import __version__
from pprint import pprint as pp
from collections import namedtuple


layer_probabilities = [1] * 3 + [2] * 2 + [3] * 1

Coefficients = namedtuple('Coefficients', 'C B Z X')



class EvogressionCreature():

    def __init__(self,
                 target_parameter: str,
                 layers: int=0,
                 generation: int=1,
                 offspring: int=0,  # indicates if creature is a result of adding previous creatures
                 full_parameter_example: dict={},
                 modifiers: dict={},
                 max_layers: int=10) -> None:
        '''
        Create creature representing a regression function with terms of form: C * (B * value + Z) ** X
        The regression function can also have multiple layers of these terms
        to create a more complex equation.
        '''
        self.layers = layers
        self.generation = generation
        self.offspring = offspring

        self.target_parameter = target_parameter
        self.full_parameter_example = full_parameter_example

        if self.layers == 0:
            self.layers = random.choice([x for x in layer_probabilities if x <= max_layers])
        self.max_layers = max_layers
        self.layer_tup = tuple(range(1, self.layers + 1))
        self.layer_str_list = [f'LAYER_{layer}' for layer in self.layer_tup]

        if modifiers == {}:
            if full_parameter_example == {}:
                print('Warning!  No modifiers or parameters provided to create EvogressionCreature!')
            self.modifiers = self.create_modifiers(layer_str_list=self.layer_str_list)
        else:
            self.modifiers = modifiers

        self.error_sum = 0


    def used_parameters(self) -> set:
        '''Return list of the parameters that are used by this creature.'''
        return {param for layer, layer_dict in self.simplify_modifiers(self.modifiers).items() for param in layer_dict
                         if param not in {'N', 'T', self.target_parameter}}
