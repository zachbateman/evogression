'''
Module containing "creatures" that each represent a potential regression equation.
'''
import random
import copy
import os
from ._version import __version__
from pprint import pprint as pp
from collections import namedtuple

try:
    from .calc_target_cython import calc_target_cython
except ImportError:
    print('\nUnable to import Cython calc_target_cython module!')
    print('If trying to install/run on a Windows computer, you may need to a C compiler.')
    print('See: https://wiki.python.org/moin/WindowsCompilers')
    print('  -> (If running Windows 7, try using Python 3.7 instead of 3.8+)\n')

try:
    from .generate_parameter_coefficients_calc import generate_parameter_coefficients_calc
except ImportError:
    print('\nUnable to import Cython generate_parameter_coefficients_calc module!')
    print('If trying to install/run on a Windows computer, you may need to a C compiler.')
    print('See: https://wiki.python.org/moin/WindowsCompilers')
    print('  -> (If running Windows 7, try using Python 3.7 instead of 3.8+)\n')


layer_probabilities = [1] * 3 + [2] * 2 + [3] * 1

Coefficients = namedtuple('Coefficients', 'C B Z X')


def fast_copy(d: dict):
    '''Used as a faster alternative to copy.deepcopy for copying to a new dict'''
    output = d.copy()
    for key, value in output.items():
        output[key] = fast_copy(value) if isinstance(value, dict) else value
    return output


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


    def create_modifiers(self, layer_str_list=None) -> dict:
        '''
        Generate and return a set of modifiers.
        The EvogressionCreature.modifiers dictionary is the definition of the equation
        created by this creature.  The dictionary contains the coefficients/parameters
        assigned to each term as well as the terms and the layers of the equation.
        '''
        # local variables for speed
        rand_rand = random.random
        rand_gauss = random.gauss
        gen_param_coeffs = generate_parameter_coefficients_calc
        full_param_example_keys = [param for param in self.full_parameter_example if param != self.target_parameter]
        parameter_usage_num = 2.5 / (len(full_param_example_keys) + 1)  # len(full_param_example) will always be >= 2

        modifiers: dict = {}
        for layer_name in layer_str_list:
            layer_modifiers = {'N': 0} if rand_rand() < 0.2 else {'N': rand_gauss(0, 0.1)}

            if layer_name != 'LAYER_1':
                layer_modifiers['T'] = Coefficients(*gen_param_coeffs())

            for param in full_param_example_keys:
                if rand_rand() < parameter_usage_num:  # resist using parameters if many of them
                    layer_modifiers[param] = Coefficients(*gen_param_coeffs())

            modifiers[layer_name] = layer_modifiers

        return modifiers


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


    def calc_target(self, parameters: dict) -> float:
        '''
        Apply the creature's modifiers to the parameters to calculate an attempt at target
        '''
        return calc_target_cython(parameters, self.modifiers)


    def mutate_to_new_creature(self, adjustments: str='fast'):
        '''
        Create a new creature based on slightly modifying this creature's modifiers.
        Rather than changing the modifiers as can happen with __add__ing the creatures,
        this merely makes small changes to the values of some of the coefficients.
        '''
        if adjustments == 'fast':
            modify_value = 0.05
        elif adjustments == 'fine':
            modify_value = 0.005

        rand_rand = random.random
        rand_gauss = random.gauss
        new_modifiers = fast_copy(self.modifiers)
        for layer_name in new_modifiers:
            if rand_rand() < 0.5:
                new_modifiers[layer_name]['N'] += rand_gauss(0, modify_value)
            for param in new_modifiers[layer_name].keys():
                if param != 'N':
                    old_coefs = new_modifiers[layer_name][param]
                    new_x = old_coefs.X
                    if rand_rand() < 0.2:
                        new_x = old_coefs.X + 1
                    elif rand_rand() < 0.2 and old_coefs.X > 1:
                        new_x = old_coefs.X - 1
                    new_coef = Coefficients(old_coefs[0] + rand_gauss(0, modify_value),
                                                          old_coefs[1] + rand_gauss(0, modify_value),
                                                          old_coefs[2] + rand_gauss(0, modify_value),
                                                          new_x)
                    new_modifiers[layer_name][param] = new_coef

        return EvogressionCreature(self.target_parameter, layers=self.layers, generation=self.generation + 1,
                                              full_parameter_example=self.full_parameter_example, modifiers=new_modifiers, max_layers=self.max_layers)


    def __add__(self, other):
        '''
        Using the __add__ ('+') operator to mate the creatures and generate offspring.
        Offspring will have a combination of modifiers from both parents that
        also includes some mutation.
        '''
        # local variables for speed
        rand_rand = random.random
        rand_tri = random.triangular
        self_modifiers = self.modifiers
        other_modifiers = other.modifiers
        new_coefficients_from_existing = _new_coefficients_from_existing
        gen_param_coeffs = generate_parameter_coefficients_calc

        # Generate new number of layers
        new_layers = int(round((self.layers + other.layers) / 2, 0))  # average (same if both are same number)

        # Possible mutation to number of layers
        if rand_rand() < 0.05:
            if new_layers > 1 and rand_rand() < 0.5:
                new_layers -= 1
            elif new_layers < max(self.max_layers, other.max_layers):
                new_layers += 1

        # Generate new, mutated coefficients through the modifier layers
        layer_names = [f'LAYER_{layer}' for layer in range(1, new_layers + 1)]
        possible_parameters = ['T'] + [key for key in self.full_parameter_example if key != self.target_parameter]
        new_modifiers = {layer_name: {'N': 0} for layer_name in layer_names}
        for layer_name in layer_names:
            reference_modifiers = [mod for mod in (self_modifiers, other_modifiers) if layer_name in mod]
            ref_mods_layer_params = set(param for mods in reference_modifiers for param in mods[layer_name])
            if not reference_modifiers:  # create new modifier layer from scratch if doesn't exist in either self or other modifiers
                # HAVE TO USE DICT LOOKUP BELOW INSTEAD OF new_modifiers_layer_name AS IT GETS ASSIGNED NEW VALUE AND LOSES REFERENCE!
                new_modifiers[layer_name] = self.create_modifiers(layer_str_list=[layer_name])[layer_name]
            else:
                new_modifiers_layer_name = new_modifiers[layer_name]
                # Calculate new 'N' value
                new_modifiers_layer_name['N'] = sum(mod[layer_name]['N'] for mod in reference_modifiers) / len(reference_modifiers) * rand_tri(0.7, 1.3, 1)
                # Calculate new Coefficients for each parameter (including 'T')
                for param in possible_parameters:
                    new_coef = None
                    if param == 'T':
                        if layer_name != 'LAYER_1':
                            new_coef = new_coefficients_from_existing(reference_modifiers, layer_name, 'T')
                            if new_coef:
                                new_modifiers_layer_name['T'] = new_coef
                    else:  # param != 'T'
                        param_in_mods = True if param in ref_mods_layer_params else False
                        if param_in_mods and rand_rand() < 0.8:  # 20% chance to not include a param in child modifiers for this layer
                            new_coef = new_coefficients_from_existing(reference_modifiers, layer_name, param)
                        elif not param_in_mods and rand_rand() < 0.1:  # chance to add new parameter to modifiers if not in parents
                            new_coef = Coefficients(*gen_param_coeffs())
                        if new_coef:
                            new_modifiers_layer_name[param] = new_coef

        return EvogressionCreature(self.target_parameter, layers=new_layers, max_layers=max(self.max_layers, other.max_layers), generation=self.generation,
                                              full_parameter_example=self.full_parameter_example, modifiers=new_modifiers,
                                              offspring=max((self.offspring, other.offspring))+1)


    def __copy__(self):
        return EvogressionCreature(self.target_parameter, layers=self.layers, max_layers=self.max_layers, generation=self.generation, modifiers=self.modifiers)


    def __repr__(self) -> str:
        printout = f'EvogressionCreature - Generation: {self.generation} - Offspring: {self.offspring}'
        for layer in self.modifiers:
            printout += f'\n  Modifiers {layer}'
            for param, coeffs in self.modifiers[layer].items():
                if param == 'N':
                    rounded_coeffs = round(coeffs, 6)
                else:
                    rounded_coeffs = {'C': round(coeffs.C, 6), 'B': round(coeffs.B, 6), 'Z': round(coeffs.Z, 6), 'X': round(coeffs.X, 6)}
                printout += f'\n     {param}: {rounded_coeffs}'
        printout += '\n'
        return printout


    def get_regression_func(self):
        return self.simplify_modifiers(self.modifiers)


    def output_python_regression_module(self, output_filename: str='regression_function', standardizer=None, directory: str='.', name_ext: str=''):
        '''Create a Python module/file with a regression function represented by this EvogressionCreature'''
        if directory != '.' and not os.path.exists(directory):
            os.mkdir(directory)

        if output_filename[-3:] == '.py':  # adding .py later; removing to easily check for periods
            output_filename = output_filename[:-3]

        output_filename = output_filename.replace('.', '_') # period in filename not valid
        output_filename = os.path.join(directory, output_filename + name_ext.replace('.', '_') + '.py')

        output_str = self.output_regression_func_as_python_module_string(standardizer=standardizer)
        with open(output_filename, 'w') as f:
            f.write(output_str)
        print('EvogressionCreature modifiers saved as regression function Python module!')


    def output_regression_func_as_python_module_string(self, standardizer=None) -> str:
        '''Create a string which creates a Python module with callable regression function.'''
        modifiers = self.simplify_modifiers(self.modifiers)
        used_parameters = {param for layer, layer_dict in modifiers.items() for param, coefficients in layer_dict.items()}

        s = f"'''\nPython regression function module generated by Evogression version {__version__}\n'''\n\n"
        s += "def regression_func(parameters: dict) -> float:\n"
        s += "    '''Generated by an EvogressionCreature'''\n\n"

        if len(used_parameters) > 0:
            if standardizer:
                s += "    # Standardize input data\n"
                s += "    standardized_data = {}\n"
                s += "    for param, value in parameters.items():\n"
                for param, modifier_dict in standardizer.data_modifiers.items():
                    if param != self.target_parameter and param in used_parameters:
                        s += f"        if param == '{param}':\n"
                        if modifier_dict['stdev'] != 0:
                            s += f"            standardized_data['{param}'] = (value - {round(modifier_dict['mean'], 6)}) / {round(modifier_dict['stdev'], 6)}\n"
                        else:
                            s += f"            standardized_data['{param}'] = value - {round(modifier_dict['mean'], 6)}\n"
                if '()' in s[-10:]:
                    s += "        pass\n"
                s += "    parameters = standardized_data\n\n"
            else:
                s += "    # No standardizer used for regression\n\n"
            s += "    T = 0  # T is the value from the previous layer if multiple layers\n\n"

        for layer in range(1, len(modifiers) + 1):
            layer_name = f'LAYER_{layer}'
            s += f"    # {layer_name}\n"
            for param, mods in modifiers[layer_name].items():
                if param == 'N':
                    s += f"    T += {round(mods, 4)}\n"
                elif param == 'T':
                    s += f"    T += {round(mods.C, 4)} * ({round(mods.B, 4)} * previous_T + {round(mods.Z, 4)}) ** {round(mods.X, 2)}\n"
                else:
                    s += f"    T += {round(mods.C, 4)} * ({round(mods.B, 4)} * parameters['{param}'] + {round(mods.Z, 4)}) ** {round(mods.X, 2)}\n"
            if layer < len(modifiers):
                s += "    previous_T, T = T, 0\n"
            s += "\n"

        if standardizer:
            s += "    # Unstandardize result\n"
            s += f"    T = T * {round(standardizer.data_modifiers[self.target_parameter]['stdev'], 6)} + {round(standardizer.data_modifiers[self.target_parameter]['mean'], 6)}\n\n"

        s += "    return T\n"
        return s



def _new_coefficients_from_existing(modifier_list: list, layer_name: str, param: str):
    '''Returns new Coefficients value from the provided data'''
    rand_tri = random.triangular  # local variables for speed

    if len(modifier_list) == 1:
        if param not in modifier_list[0][layer_name]:
            return None
        else:
            c1 = modifier_list[0][layer_name][param]
            new_x = round(c1[3] * rand_tri(0.7, 1.3, 1), 0)
            return Coefficients(c1[0] * rand_tri(0.7, 1.3, 1), c1[1] * rand_tri(0.7, 1.3, 1), c1[2] * rand_tri(0.7, 1.3, 1), new_x if new_x >= 0 else 0)
    else:  # two modifiers
        try:
            c1, c2 = modifier_list[0][layer_name][param], modifier_list[1][layer_name][param]
            new_x = round((c1[3] + c2[3]) / 2 * rand_tri(0.7, 1.3, 1), 0)
            return Coefficients((c1[0]  + c2[0]) / 2, (c1[1] + c2[1]) / 2, (c1[2] + c2[2]) / 2, new_x if new_x >= 0 else 0)
        except KeyError:  # if param not in both sets of modifiers, recursively find coefficients of modifier set with param
            coef_1, coef_2 = _new_coefficients_from_existing([modifier_list[0]], layer_name, param), _new_coefficients_from_existing([modifier_list[1]], layer_name, param)
            return coef_1 if coef_1 else coef_2
