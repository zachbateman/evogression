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
    print('Calculations will run slower...')
    print('  -> (If running Windows 7, try using Python 3.7 instead of 3.8+)\n')

try:
    from .generate_parameter_coefficients_calc import generate_parameter_coefficients_calc
except ImportError:
    print('\nUnable to import Cython generate_parameter_coefficients_calc module!')
    print('Calculations will run slower...')
    print('  -> (If running Windows 7, try using Python 3.7 instead of 3.8+)\n')


layer_probabilities = [1] * 5 + [2] * 3 + [3] * 2 + [4] * 1

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
                 max_layers=None) -> None:
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
            if max_layers and max_layers > 0:
                self.layers = random.choice([x for x in layer_probabilities if x <= max_layers])
            else:
                self.layers = random.choice(layer_probabilities)
        self.max_layers = max_layers
        self.layer_tup = tuple(range(1, self.layers + 1))
        self.layer_str_list = [f'LAYER_{layer}' for layer in self.layer_tup]

        if modifiers == {}:
            if full_parameter_example == {}:
                print('Warning!  No modifiers or parameters provided to create EvogressionCreature!')
            self.modifiers = self.create_initial_modifiers()
        else:
            self.modifiers = modifiers

        self.error_sum = 0


    def create_initial_modifiers(self) -> dict:
        '''
        Generate and return a set of modifiers.
        The EvogressionCreature.modifiers dictionary is the definition of the equation
        created by this creature.  The dictionary contains the coefficients/parameters
        assigned to each term as well as the terms and the layers of the equation.
        '''
        parameter_usage_num = 2.5 / (len(self.full_parameter_example) + 1)  # local variables for speed
        rand_rand = random.random
        rand_gauss = random.gauss
        targ_param = self.target_parameter
        gen_param_coeffs = self.generate_parameter_coefficients
        full_param_example_keys = self.full_parameter_example.keys()

        modifiers: dict = {}
        for layer_name in self.layer_str_list:
            layer_modifiers = {'N': 0} if rand_rand() < 0.2 else {'N': rand_gauss(0, 0.1)}
            for param in full_param_example_keys:
                # resist using parameters if many of them
                # len(full_param_example) will always be >= 2
                if rand_rand() < parameter_usage_num and param != targ_param:
                    C, B, Z, X = gen_param_coeffs()
                    if X != 0:  # 0 exponent makes term overly complex for value added; don't include
                        layer_modifiers[param] = Coefficients(C, B, Z, X)
                    else:
                        layer_modifiers['N'] += C
            if layer_name != 'LAYER_1':
                C, B, Z, X = gen_param_coeffs()
                if X == 0:  # want every layer > 1 to include a T term!!
                    X = 1
                layer_modifiers['T'] = Coefficients(C, B, Z, X)

            modifiers[layer_name] = layer_modifiers

        return modifiers


    def generate_parameter_coefficients(self):
        try:  # optimize for cython-available case
            return generate_parameter_coefficients_calc()
        except NameError:
            return self._generate_parameter_coefficients()


    def _generate_parameter_coefficients(self):
        '''
        Create randomized coefficients/parameters for that can
        be assigned to a single term in the modifiers dict.
        '''
        rand_rand = random.random  # local for speed
        rand_tri = random.triangular  # local for speed
        C = 1 if rand_rand() < 0.4 else rand_tri(0, 2, 1)
        B = 1 if rand_rand() < 0.3 else rand_tri(0, 2, 1)
        Z = 0 if rand_rand() < 0.4 else rand_tri(-2, 2, 0)
        if rand_rand() < 0.5:
            C = -C
        if rand_rand() < 0.5:
            B = -B
        X = 1 if rand_rand() < 0.4 else random.choice([0, 2, 2, 2, 2, 2, 3, 3])
        return C, B, Z, X


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
            for param, param_dict in layer_dict.items():

                # don't want a 'T' parameter in first layer as no previous result to operate on
                if layer == 'LAYER_1' and param == 'T':
                    del new_modifiers[layer]['T']

                # if 'T' element is raised to 0 power... previous layer(s) are not used.
                # Rebuild the layers without the unused ones before the current layer
                if current_layer_num > 1 and param == 'T' and param_dict.X == 0:
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
        try:
            return calc_target_cython(parameters, self.modifiers, self.layer_str_list)
        except:  # if cython extension not available
            T = None  # has to be None on first layer
            for layer in self.layer_tup:
                T = self._calc_single_layer_target(parameters, layer, previous_T=T)
            return T


    def _calc_single_layer_target(self, parameters: dict, layer: int, previous_T=None) -> float:
        '''
        Apply creature's modifiers to parameters of ONE LAYER to calculate or help calculate target.
        THIS IS THE MOST EXPENSIVE PART OF EVOGRESSION!!!

        This is now only BACKUP to Cython implementation! (over twice as fast)
        '''
        T = 0
        layer_modifiers = self.modifiers[f'LAYER_{layer}']

        def param_value_component(layer_modifiers: dict, param: str, value: float) -> float:
            try:
                mods = layer_modifiers[param]
                return mods.C * (mods.B * value + mods.Z) ** mods.X
            except KeyError:  # if param is not in self.modifiers[layer_name]
                return 0
            except OverflowError:
                return 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

        for param, value in parameters.items():
            T += param_value_component(layer_modifiers, param, value)
        if previous_T:
            T += param_value_component(layer_modifiers, 'T', previous_T)

        try:
            T += layer_modifiers['N']
        except KeyError:
            pass

        return T


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
        rand_rand = random.random  # local variable for speed
        rand_tri = random.triangular

        # Generate new number of layers
        new_layers = int(round((self.layers + other.layers) / 2, 0))  # average (same if both are same number)
        # Possible mutation to number of layers
        if rand_rand() < 0.05:
            if new_layers > 1 and rand_rand() < 0.5:
                new_layers -= 1
            elif new_layers < max(self.max_layers, other.max_layers):
                new_layers += 1

        def create_new_coefficients(new_modifiers: dict, modifier_list: list, layer_name: str) -> None:
            '''Modifies new_modifiers in place'''
            if len(modifier_list) == 1:
                c1 = modifier_list[0][layer_name][param]
                new_x = round(c1[3] * rand_tri(0.7, 1.3, 1), 0)
                new_modifiers[layer_name][param] = Coefficients(c1[0] * rand_tri(0.7, 1.3, 1), c1[1] * rand_tri(0.7, 1.3, 1), c1[2] * rand_tri(0.7, 1.3, 1), new_x if new_x >= 0 else 0)
            else:  # two modifiers
                c1, c2 = modifier_list[0][layer_name][param], modifier_list[1][layer_name][param]
                new_x = round((c1[3] + c2[3]) / 2 * rand_tri(0.7, 1.3, 1), 0)
                new_modifiers[layer_name][param] = Coefficients((c1[0]  + c2[0]) / 2, (c1[1] + c2[1]) / 2, (c1[2] + c2[2]) / 2, new_x if new_x >= 0 else 0)


        possible_parameters = ['N', 'T'] + [key for key in self.full_param_example if key != self.target_parameter]
        new_modifiers = {f'LAYER_{layer}': {'N': 0} for layer in range(1, new_layers + 1)}

        self_modifiers = self.modifiers  # local variable for speed
        other_modifiers = other.modifiers  # local variable for speed

        for layer in range(1, new_layers + 1):
            layer_name = f'LAYER_{layer}'
            for param in possible_parameters:
                if param == 'N':
                    if layer_name in self_modifiers and layer_name in other_modifiers:
                        if param in self_modifiers[layer_name] and param in other_modifiers[layer_name]:
                            new_N = (self_modifiers[layer_name][param] + other_modifiers[layer_name][param]) / 2
                            new_N *= rand_tri(0.7, 1.3, 1)
                            new_modifiers[layer_name]['N'] = new_N
                        elif param in self_modifiers[layer_name]:
                            new_N = self_modifiers[layer_name][param] * rand_tri(0.7, 1.3, 1)
                            new_modifiers[layer_name]['N'] = new_N
                        elif param in other_modifiers[layer_name]:
                            new_N = other_modifiers[layer_name][param] * rand_tri(0.7, 1.3, 1)
                            new_modifiers[layer_name]['N'] = new_N
                    elif layer_name in self_modifiers:
                        new_N = self_modifiers[layer_name][param] * rand_tri(0.7, 1.3, 1)
                        new_modifiers[layer_name]['N'] = new_N
                    elif layer_name in other_modifiers:
                        try:
                            new_N = other_modifiers[layer_name][param]
                        except KeyError:
                            breakpoint()
                        new_N *= rand_tri(0.7, 1.3, 1)
                        new_modifiers[layer_name]['N'] = new_N

                else:  # param is one of ['T', 'B', 'C', 'X', 'Z']
                    force_keep_param = False if param != 'T' or layer == 1 else True
                    if layer_name in self_modifiers and layer_name in other_modifiers:
                        if param != 'T' or layer > 1:
                            if param in self_modifiers[layer_name] and param in other_modifiers[layer_name]:
                                create_new_coefficients(new_modifiers, [self_modifiers, other_modifiers], layer_name)

                        elif param in self_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [self_modifiers], layer_name)

                        elif param in other_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [other_modifiers], layer_name)

                    elif layer_name in self_modifiers:
                        if param in self_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [self_modifiers], layer_name)

                    elif layer_name in other_modifiers:
                        if param in other_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [other_modifiers], layer_name)

        # Chance to add or remove parameter modifiers
        remove_modifiers = []
        add_modifiers = []
        for layer in range(1, new_layers + 1):
            layer_name = f'LAYER_{layer}'
            for param, values in new_modifiers[layer_name].items():
                if rand_rand() < 0.01 and param != 'T':
                    remove_modifiers.append((layer_name, param))
            for param in possible_parameters:
                if param not in new_modifiers[layer_name]:
                    if rand_rand() < 0.01 or (layer > 1 and param == 'T'):
                        add_modifiers.append((layer_name, param))

        for remove_tup in remove_modifiers:
            if remove_tup[1] != 'N':
                del new_modifiers[remove_tup[0]][remove_tup[1]]
            else:
                new_modifiers[remove_tup[0]][remove_tup[1]] = 0

        for add_tup in add_modifiers:
            if add_tup[1] == 'N':
                new_modifiers[add_tup[0]]['N'] = 0 if rand_rand() < 0.2 else random.gauss(0, 1)
            else:
                # C, B, Z, X = self.generate_parameter_coefficients()
                # new_modifiers[add_tup[0]][add_tup[1]] = {'C': C, 'B': B, 'Z': Z, 'X': X}
                new_modifiers[add_tup[0]][add_tup[1]] = Coefficients(*self.generate_parameter_coefficients())

        try:
            new_max_layers = max((self.max_layers, other.max_layers))
        except TypeError:  # if one has None as max_layers
            new_max_layers = None

        return EvogressionCreature(self.target_parameter, layers=new_layers, max_layers=new_max_layers, generation=self.generation,
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
