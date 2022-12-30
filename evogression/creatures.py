'''
Module containing "creatures" that each represent a potential regression equation.
'''
import random
import copy
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


    def simplify_modifiers(self, modifiers: dict={}) -> dict:
        '''
        Analyzes the mathematics of self.modifiers and simplifies if possible
        '''
        if modifiers == {}:
            old_modifiers, new_modifiers = self.modifiers, copy.deepcopy(self.modifiers)
        else:
            old_modifiers, new_modifiers = modifiers, copy.deepcopy(modifiers)

        new_base_layer = 0  # will be converted to the new LAYER_1 if any layers are deleted
        for layer, layer_dict in old_modifiers.items():
            current_layer_num = int(layer[-1])
            for param, param_coef in layer_dict.items():

                # don't want a 'T' parameter in first layer as no previous result to operate on
                if layer == 'LAYER_1' and param == 'T':
                    del new_modifiers[layer]['T']

                # if 'T' element is raised to 0 power... previous layer(s) are not used.
                # Rebuild the layers without the unused ones before the current layer
                if current_layer_num > 1 and param == 'T' and param_coef.X == 0:
                    new_base_layer = current_layer_num

            if current_layer_num > 1 and 'T' not in layer_dict:
                if not new_base_layer:
                    new_base_layer = current_layer_num
                elif current_layer_num > new_base_layer:
                    new_base_layer = current_layer_num

        # remove unused first layer(s)
        if new_base_layer > 0:
            self.layers = len(new_modifiers) - new_base_layer + 1  # RESET THIS CREATURE'S LAYERS!!!
            for i in range(1, new_base_layer):
                del new_modifiers[f'LAYER_{i}']
            for i in range(1, self.layers + 1):
                new_modifiers[f'LAYER_{i}'] = new_modifiers.pop(f'LAYER_{new_base_layer + i - 1}')

        for i in range(self.layers + 1, len(old_modifiers) + 1):
            try:
                del new_modifiers[f'LAYER_{i}']
            except KeyError:
                pass

        self.layers = len(new_modifiers)
        if new_modifiers != old_modifiers and self.layers != len(new_modifiers):
            print(f'self.layers: {self.layers}   new_modifiers: {len(new_modifiers)}')
            pp(old_modifiers)
            pp(new_modifiers)

        return new_modifiers


    def get_regression_func(self) -> dict:
        return self.simplify_modifiers(self.modifiers)
