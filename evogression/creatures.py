'''
Module containing "creatures" that each represent a potential regression equation.
'''
import random
import copy
import warnings
from pprint import pprint as pp


layer_probabilities = [1] * 2 + [2] * 4 + [3] * 5 + [4] * 3 + [5] * 2 + [6] * 1

class EvogressionCreature():

    def __init__(self,
                       target_parameter: str,
                       layers: int=0,
                       hunger: int=100,
                       generation: int=1,
                       mutability: float=0,
                       full_parameter_example: dict={},
                       no_negative_exponents: bool=True,
                       modifiers: dict={}) -> None:

        # hunger decreases over time proportional to this creature's regression complexity
        # successfully "eating" a sample will increase self.hunger
        # creature dies when self.hunger == 0
        self.hunger = hunger
        self.layers = layers
        self.generation = generation
        self.mutability = 0.1 * (random.random() + 0.5) if mutability == 0 else mutability

        self.no_negative_exponents = no_negative_exponents
        self.target_parameter = target_parameter
        self.full_parameter_example = full_parameter_example

        # for any given input, want to modify/evolve as following:
        # inputs = {'a': 5.7, 'b': 3.2, 'c': 4.3}
        # target = 17.9
        # C_1 * (B_1 * a + Z_1) ** X_1 + C_2 * (B_2 * b + Z_2) ** X_2 + ...  + N_1 = T_1
        # T_1 = target (17.9) in single layer, else, feed T_1 as additional arg into next layer, etc.

        if modifiers == {}:
            if full_parameter_example == {}:
                print('Warning!  No modifiers or parameters provided to create EvogressionCreature!')
            self.modifiers = self.create_initial_modifiers()
        else:
            self.modifiers = modifiers


    def create_initial_modifiers(self) -> dict:
        # creates initial modifiers used with each given parameter
        if self.layers == 0:
            self.layers = random.choice(layer_probabilities)

        modifiers = {}
        for layer in range(1, self.layers + 1):
            modifiers[f'LAYER_{layer}'] = {}
            modifiers[f'LAYER_{layer}']['N'] = 0 if random.random() < 0.2 else random.gauss(0, 3 * self.mutability)
            for param in self.full_parameter_example.keys():
                # resist using parameters if many of them
                # len(full_param_example) will always be >= 2
                if random.random() < (1 / len(self.full_parameter_example)) + 0.3 and param != self.target_parameter:
                    C, B, Z, X = self.generate_parameter_coefficients()
                    if X != 0:  # 0 exponent makes term overly complex for value added; don't include
                        modifiers[f'LAYER_{layer}'][param] = {'C': C, 'B': B, 'Z': Z, 'X': X}
                    else:
                        modifiers[f'LAYER_{layer}']['N'] += C
            if layer > 1:
                C, B, Z, X = self.generate_parameter_coefficients()
                if X == 0:  # want every layer > 1 to include a T term!!
                    X = 1
                modifiers[f'LAYER_{layer}']['T'] = {'C': C, 'B': B, 'Z': Z, 'X': X}

        return modifiers

    def generate_parameter_coefficients(self):
        C = 1 if random.random() < 0.4 else random.gauss(1, self.mutability)
        B = 1 if random.random() < 0.3 else random.gauss(1, 2 * self.mutability)
        Z = 0 if random.random() < 0.4 else random.gauss(0, 3 * self.mutability)
        if self.no_negative_exponents:
            X = 1 if random.random() < 0.4 else random.choice([0] * 1 + [2] * 5 + [3] * 2)
        else:
            X = 1 if random.random() < 0.4 else random.choice([-2] * 1 + [-1] * 5 + [0] * 3 + [2] * 5 + [3] * 1)
        return C, B, Z, X

    def simplify_modifiers(self, modifiers: dict={}) -> dict:
        '''Method looks at the mathematics of self.modifiers and simplifies if possible'''
        if modifiers == {}:
            old_modifiers = self.modifiers
            new_modifiers = copy.deepcopy(self.modifiers)
        else:
            old_modifiers = modifiers
            new_modifiers = copy.deepcopy(modifiers)

        new_base_layer = False  # will be converted to the new LAYER_1 if any layers are deleted
        for layer, layer_dict in old_modifiers.items():
            current_layer_num = int(layer[-1])
            for param, param_dict in layer_dict.items():

                # don't want a 'T' parameter in first layer as no previous result to operate on
                if layer == 'LAYER_1' and param == 'T':
                    del new_modifiers[layer]['T']

                # if 'T' element is raised to 0 power... previous layer(s) are not used.
                # Rebuild the layers without the unused ones before the current layer
                if current_layer_num > 1 and param == 'T' and param_dict['X'] == 0:
                    new_base_layer = current_layer_num

            if current_layer_num > 1 and 'T' not in layer_dict:
                if not new_base_layer:
                    new_base_layer = current_layer_num
                elif current_layer_num > new_base_layer:
                    new_base_layer = current_layer_num

        # remove unused first layer(s)
        if new_base_layer:
            # RESET THIS CREATURE'S LAYERS!!!
            self.layers = len(new_modifiers) - new_base_layer + 1

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
        '''Apply the creature's modifiers to the parameters to calculate an attempt at target'''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            T = None  # has to be None on first layer
            # for layer in range(1, self.layers + 1):  # NOT SURE WHY SOMETIMES self.layers != len(self.modifiers)!!!
            for layer in range(1, len(self.modifiers) + 1):
                T = self._calc_single_layer_target(parameters, layer, previous_T=T)
            return T

    def _calc_single_layer_target(self, parameters: dict, layer: int, previous_T=None) -> float:
        '''Apply creature's modifiers to parameters of ONE LAYER to calculate or help calculate target'''
        T = 0
        layer_modifiers = self.modifiers[f'LAYER_{layer}']
        for param, value in parameters.items():
            try:
                mods = layer_modifiers[param]
                T += mods['C'] * (mods['B'] * value + mods['Z']) ** mods['X']
            except KeyError:  # if param is not in self.modifiers[layer_name]
                pass
            except ZeroDivisionError:  # could occur if exponent is negative
                pass
            except OverflowError:
                return 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

        if previous_T:
            try:
                mods = layer_modifiers['T']
                T += mods['C'] * (mods['B'] * previous_T + mods['Z']) ** mods['X']
            except KeyError:  # if 'T' is somehow not in self.modifiers[layer_name] (likely due to mating mutation?)
                pass
            except ZeroDivisionError:  # could occur if exponent is negative
                pass
            except OverflowError:
                return 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

        try:
            T += layer_modifiers['N']
        except KeyError:
            pass

        return T

    @property
    def complexity_cost(self):
        '''
        Calculate an int to reduce hunger by based on the complexity of this creature.
        The idea is to penalize more complex models and thereby create a tendancy
        to develop a simpler model.
        '''
        cost = 10
        cost += 3 * self.layers  # cost will be AT LEAST 3
        cost += int(round(sum(len(layer_dict) / 5 for layer_dict in self.modifiers.values()), 0))
        return cost

    def __add__(self, other):
        '''
        Using the __add__ ('+') operator to mate the creatures and generate offspring.
        Offspring will have a combination of modifiers from both parents that
        also includes some mutation.
        '''
        combined_hunger = self.hunger + other.hunger
        chance_of_mating = (combined_hunger - 75) / 100

        if random.random() < (1 - chance_of_mating):
            return None

        # Generate new number of layers
        if self.layers == other.layers:
            new_layers = int(self.layers)
        elif abs(self.layers - other.layers) < 1:
            new_layers = int(self.layers) if random.random() < 0.5 else int(other.layers)
        else:
            new_layers = int(round((self.layers + other.layers) / 2, 0))

        if random.random() < 0.05:  # mutation to number of layers
            if random.random() < 0.5 and new_layers > 1:
                new_layers -= 1
            else:
                new_layers += 1

        new_generation = max(self.generation, other.generation) + 1

        def mutate_multiplier(mutability) -> float:
            return 1 + mutability * (random.random() - 0.5) / 25

        # Generate new mutability
        new_mutability = (self.mutability + other.mutability) / 2
        new_mutability *= 3 * mutate_multiplier(new_mutability)
        if new_mutability > 30:  # HAVE TO LIMIT MUTABILITY OR EVOLUTION BECOMES UNSTABLE AND THROWS ERRORS!!!
            new_mutability = 30
        elif new_mutability < 0:  # mutability can't be negative!!!
            new_mutability = 0.001

        # Generate new modifier layer(s) based on self and other
        def get_possible_parameters(full_param_example, target_parameter):
            pos_params = ['N', 'T']
            pos_params.extend(sorted(key for key in full_param_example if key != target_parameter))
            return pos_params

        possible_parameters = get_possible_parameters(self.full_parameter_example, self.target_parameter)
        # possible_parameters = ['N', 'T'].extend(sorted([key for key in self.full_parameter_example if key != self.target_parameter]))
        # breakpoint()
        coefficients = ['C', 'B', 'Z', 'X']
        new_modifiers = {f'LAYER_{layer}': {'N': 0} for layer in range(1, new_layers + 1)}
        for layer in range(1, new_layers + 1):
            layer_name = f'LAYER_{layer}'
            for param in possible_parameters:
                if param == 'N':
                    if layer_name in self.modifiers and layer_name in other.modifiers:
                        if param in self.modifiers[layer_name] and param in other.modifiers[layer_name]:
                            new_N = (self.modifiers[layer_name][param] + other.modifiers[layer_name][param]) / 2
                            new_N *= mutate_multiplier(new_mutability)
                            new_modifiers[layer_name]['N'] = new_N
                        elif param in self.modifiers[layer_name]:
                            new_N = self.modifiers[layer_name][param] * mutate_multiplier(new_mutability)
                            new_modifiers[layer_name]['N'] = new_N
                        elif param in other.modifiers[layer_name]:
                            new_N = other.modifiers[layer_name][param] * mutate_multiplier(new_mutability)
                            new_modifiers[layer_name]['N'] = new_N
                    elif layer_name in self.modifiers:
                        new_N = self.modifiers[layer_name][param] * mutate_multiplier(new_mutability)
                        new_modifiers[layer_name]['N'] = new_N
                    elif layer_name in other.modifiers:
                        try:
                            new_N = other.modifiers[layer_name][param]
                        except KeyError:
                            breakpoint()
                        new_N *= mutate_multiplier(new_mutability)
                        new_modifiers[layer_name]['N'] = new_N

                else:  # param is one of ['T', 'B', 'C', 'X', 'Z']
                    if not (param == 'T' and layer == 1):
                        if layer_name in self.modifiers and layer_name in other.modifiers:
                            if param in self.modifiers[layer_name] and param in other.modifiers[layer_name]:
                                new_modifiers[layer_name][param] = {}
                                for coef in coefficients:
                                    new_coef = (self.modifiers[layer_name][param][coef] + other.modifiers[layer_name][param][coef]) / 2
                                    if coef == 'X':
                                        new_coef = round(new_coef * mutate_multiplier(new_mutability), 0)
                                        if self.no_negative_exponents and new_coef < 0:
                                            new_coef = 0
                                    else:
                                        new_coef *= mutate_multiplier(new_mutability)
                                    new_modifiers[layer_name][param][coef] = new_coef
                            elif param in self.modifiers[layer_name] and random.random() < 0.5:
                                new_modifiers[layer_name][param] = {}
                                for coef in coefficients:
                                    new_coef = self.modifiers[layer_name][param][coef]
                                    if coef == 'X':
                                        new_coef = round(new_coef * mutate_multiplier(new_mutability), 0)
                                        if self.no_negative_exponents and new_coef < 0:
                                            new_coef = 0
                                    else:
                                        new_coef *= mutate_multiplier(new_mutability)
                                    new_modifiers[layer_name][param][coef] = new_coef
                            elif param in other.modifiers[layer_name] and random.random() < 0.5:
                                new_modifiers[layer_name][param] = {}
                                for coef in coefficients:
                                    new_coef = other.modifiers[layer_name][param][coef]
                                    if coef == 'X':
                                        new_coef = round(new_coef * mutate_multiplier(new_mutability), 0)
                                        if self.no_negative_exponents and new_coef < 0:
                                            new_coef = 0
                                    else:
                                        new_coef *= mutate_multiplier(new_mutability)
                                    new_modifiers[layer_name][param][coef] = new_coef
                        elif layer_name in self.modifiers:
                            if param in self.modifiers[layer_name] and random.random() < 0.5:
                                new_modifiers[layer_name][param] = {}
                                for coef in coefficients:
                                    new_coef = self.modifiers[layer_name][param][coef]
                                    if coef == 'X':
                                        new_coef = round(new_coef * mutate_multiplier(new_mutability), 0)
                                        if self.no_negative_exponents and new_coef < 0:
                                            new_coef = 0
                                    else:
                                        new_coef *= mutate_multiplier(new_mutability)
                                    new_modifiers[layer_name][param][coef] = new_coef
                        elif layer_name in other.modifiers:
                            if param in other.modifiers[layer_name] and random.random() < 0.5:
                                new_modifiers[layer_name][param] = {}
                                for coef in coefficients:
                                    new_coef = other.modifiers[layer_name][param][coef]
                                    if coef == 'X':
                                        new_coef = round(new_coef * mutate_multiplier(new_mutability), 0)
                                        if self.no_negative_exponents and new_coef < 0:
                                            new_coef = 0
                                    else:
                                        new_coef *= mutate_multiplier(new_mutability)
                                    new_modifiers[layer_name][param][coef] = new_coef

        # Chance to add or remove parameter modifiers
        remove_modifiers = []
        add_modifiers = []
        for layer in range(1, new_layers + 1):
            layer_name = f'LAYER_{layer}'
            for param, values in new_modifiers[layer_name].items():
                if random.random() < 0.01 * new_mutability:
                    remove_modifiers.append((layer_name, param))
            for param in possible_parameters:
                if param not in new_modifiers[layer_name]:
                    if random.random() < 0.01 * new_mutability:
                        add_modifiers.append((layer_name, param))

        for remove_tup in remove_modifiers:
            if remove_tup[1] != 'N':
                del new_modifiers[remove_tup[0]][remove_tup[1]]
            else:
                new_modifiers[remove_tup[0]][remove_tup[1]] = 0

        for add_tup in add_modifiers:
            if add_tup[1] == 'N':
                new_modifiers[add_tup[0]]['N'] = 0 if random.random() < 0.2 else random.gauss(0, new_mutability)
            else:
                C, B, Z, X = self.generate_parameter_coefficients()
                new_modifiers[add_tup[0]][add_tup[1]] = {'C': C, 'B': B, 'Z': Z, 'X': X}

        if combined_hunger > 150:
            self.hunger -= 25
            other.hunger -= 25
        elif combined_hunger > 100:
            self.hunger -= 10
            other.hunger -= 10

        return EvogressionCreature(self.target_parameter, layers=new_layers, hunger=100, generation=new_generation, mutability=new_mutability, full_parameter_example=self.full_parameter_example, modifiers=new_modifiers)

    def __copy__(self):
        return EvogressionCreature(self.target_parameter, layers=self.layers, hunger=self.hunger, generation=self.generation, mutability=self.mutability, modifiers=self.modifiers)

    def __repr__(self) -> str:
        printout = f'EvogressionCreature - Generation: {self.generation}'
        for layer in self.modifiers:
            printout += f'\n  Modifiers {layer}'
            for param, coeffs in self.modifiers[layer].items():
                printout += f'\n     {param}: {coeffs}'
        printout += '\n'
        return printout

    def get_regression_func(self):
        return self.simplify_modifiers(self.modifiers)


    def output_python_regression_module(self, output_filename: str='regression_function.py'):
        '''Create a Python module/file with a regression function represented by this EvogressionCreature'''
        output_str = self.output_regression_func_as_python_module_string()
        with open(output_filename, 'w') as f:
            f.write(output_str)
        print(f'EvogressionCreature modifiers outputted as regression function Python module!')


    def output_regression_func_as_python_module_string(self) -> str:
        '''Create a string which creates a Python module with callable regression function.'''

        modifiers = self.simplify_modifiers(self.modifiers)

        s = "'''\nPython regression function module generated by Evogression.\n'''\n\n"
        s += "def regression_func(parameters: dict) -> float:\n"
        s += "    '''Generated by an Evogression creature'''\n\n"
        s += "    T = 0  # T is the value from the previous layer if multiple layers\n\n"

        for layer in range(1, len(modifiers) + 1):
            layer_name = f'LAYER_{layer}'

            s += f"    # {layer_name}\n"
            for param, mods in modifiers[layer_name].items():
                if param == 'N':
                    s += f"    T += {round(mods, 4)}\n"
                elif param == 'T':
                    s += f"    T += {round(mods['C'], 4)} * ({round(mods['B'], 4)} * previous_T + {round(mods['Z'], 4)}) ** {round(mods['X'], 2)}\n"
                else:
                    s += f"    T += {round(mods['C'], 4)} * ({round(mods['B'], 4)} * parameters['{param}'] + {round(mods['Z'], 4)}) ** {round(mods['X'], 2)}\n"
            if layer < len(modifiers):
                s += "    previous_T, T = T, 0\n"
            s += "\n"
        s += "    return T\n"

        return s
