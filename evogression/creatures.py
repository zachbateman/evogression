'''
Module containing "creatures" that each represent a potential regression equation.
'''
import random
import copy
from pprint import pprint as pp
try:
    from . import calc_target_cython
    cython_available = True
except ImportError:
    print('\nUnable to import Cython calc_target_cython module!')
    print('Calculations will run significantly slower...\n')
    cython_available = False

try:
    from . import generate_parameter_coefficients_calc
    param_coeff_cython_available = True
except ImportError:
    print('\nUnable to import Cython generate_parameter_coefficients_calc module!')
    print('Calculations will run slightly slower...\n')
    param_coeff_cython_available = False


random.seed(1000)
layer_probabilities = [1] * 5 + [2] * 3 + [3] * 2 + [4] * 1

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
        '''
        Create creature representing a regression function with terms of form: C * (B * value + Z) ** X
        The regression function can also have multiple layers of these terms.
        '''
        self.hunger = hunger  # creature dies when self.hunger <= 0
        self.layers = layers
        self.generation = generation
        self.mutability = 0.1 * (random.random() + 0.5) if mutability == 0 else mutability

        self.no_negative_exponents = no_negative_exponents
        self.target_parameter = target_parameter
        self.full_parameter_example = full_parameter_example

        if self.layers == 0:
            self.layers = random.choice(layer_probabilities)
        self.layer_list = list(range(1, self.layers + 1))
        self.layer_str_list = [f'LAYER_{layer}' for layer in self.layer_list]

        if modifiers == {}:
            if full_parameter_example == {}:
                print('Warning!  No modifiers or parameters provided to create EvogressionCreature!')
            self.modifiers = self.create_initial_modifiers()
        else:
            self.modifiers = modifiers

        self.modifier_hash = hash(repr(self.modifiers))  # used for caching purposes!


    def create_initial_modifiers(self) -> dict:

        # local variables for speed
        parameter_usage_num = 2.5 / (len(self.full_parameter_example) + 1)
        rand_rand = random.random
        # rand_gauss = random.gauss
        rand_tri = random.triangular
        targ_param = self.target_parameter
        gen_param_coeffs = self.generate_parameter_coefficients
        full_param_example_keys = self.full_parameter_example.keys()

        modifiers: dict = {}
        for layer_name in self.layer_str_list:
            modifiers[layer_name] = {}
            # modifiers[f'LAYER_{layer}']['N'] = 0 if rand_rand() < 0.2 else random.gauss(0, 3 * self.mutability)
            modifiers[layer_name]['N'] = 0 if rand_rand() < 0.2 else rand_tri(-9 * self.mutability, 0, 9 * self.mutability)
            for param in full_param_example_keys:
                # resist using parameters if many of them
                # len(full_param_example) will always be >= 2
                if rand_rand() < parameter_usage_num and param != targ_param:
                    C, B, Z, X = gen_param_coeffs()
                    if X != 0:  # 0 exponent makes term overly complex for value added; don't include
                        modifiers[layer_name][param] = {'C': C, 'B': B, 'Z': Z, 'X': X}
                    else:
                        modifiers[layer_name]['N'] += C
            if layer_name != 'LAYER_1':
                C, B, Z, X = gen_param_coeffs()
                if X == 0:  # want every layer > 1 to include a T term!!
                    X = 1
                modifiers[layer_name]['T'] = {'C': C, 'B': B, 'Z': Z, 'X': X}

        return modifiers


    def generate_parameter_coefficients(self):

        try:  # optimize for cython-available case
            return generate_parameter_coefficients_calc.generate_parameter_coefficients_calc(self.mutability, self.no_negative_exponents)
        except NameError:
            return self._generate_parameter_coefficients()


    def _generate_parameter_coefficients(self):
        rand_rand = random.random  # local variable for speed
        rand_tri = random.triangular
        rand_choice = random.choice
        mutability = self.mutability  # local variable for speed
        C = 1 if rand_rand() < 0.4 else rand_tri(-3 * mutability, 1, 3 * mutability)
        B = 1 if rand_rand() < 0.3 else rand_tri(-6 * mutability, 1, 6 * mutability)
        Z = 0 if rand_rand() < 0.4 else rand_tri(-9 * mutability, 0, 9 * mutability)
        if rand_rand() < 0.5:
            C = -C
        if rand_rand() < 0.5:
            B = -B
        if self.no_negative_exponents:
            X = 1 if rand_rand() < 0.4 else rand_choice([0] * 1 + [2] * 5 + [3] * 2)
        else:
            X = 1 if rand_rand() < 0.4 else rand_choice([-2] * 1 + [-1] * 5 + [0] * 3 + [2] * 5 + [3] * 1)
        return C, B, Z, X


    def used_parameters(self) -> set:
        '''Return list of the parameters that are used by this creature.'''
        return {param for layer, layer_dict in self.simplify_modifiers(self.modifiers).items() for param in layer_dict if param not in {'N', 'T', self.target_parameter}}


    def simplify_modifiers(self, modifiers: dict={}) -> dict:
        '''Method looks at the mathematics of self.modifiers and simplifies if possible'''
        if modifiers == {}:
            old_modifiers = self.modifiers
            new_modifiers = copy.deepcopy(self.modifiers)
        else:
            old_modifiers = modifiers
            new_modifiers = copy.deepcopy(modifiers)

        new_base_layer = 0  # will be converted to the new LAYER_1 if any layers are deleted
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
            return calc_target_cython.calc_target_cython(parameters, self.modifiers, self.layer_str_list)
        except:  # if cython extension not available
            T = None  # has to be None on first layer
            for layer in self.layer_list:
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
                return mods['C'] * (mods['B'] * value + mods['Z']) ** mods['X']
            except KeyError:  # if param is not in self.modifiers[layer_name]
                return 0
            except ZeroDivisionError:  # could occur if exponent is negative
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


    @property
    def complexity_cost(self):
        '''
        Calculate an int to reduce hunger by based on the complexity of this creature.
        The idea is to penalize more complex models and thereby create a tendency
        to develop a simpler model.
        '''
        try:
            return self._complexity_cost
        except:
            self._complexity_cost = 15 + int(round(sum(len(layer_dict) for layer_dict in self.modifiers.values()) / 4, 0))
            return self._complexity_cost


    def __add__(self, other):
        '''
        Using the __add__ ('+') operator to mate the creatures and generate offspring.
        Offspring will have a combination of modifiers from both parents that
        also includes some mutation.
        '''
        rand_rand = random.random  # local variable for speed
        self_layers, other_layers = self.layers, other.layers

        combined_hunger = self.hunger + other.hunger
        chance_of_mating = (combined_hunger - 100) / 100

        if rand_rand() < (1 - chance_of_mating):
            return None

        # Generate new number of layers
        if self_layers == other_layers:
            new_layers = int(self_layers)
        elif abs(self_layers - other_layers) < 1:
            new_layers = int(self_layers) if rand_rand() < 0.5 else int(other_layers)
        else:
            new_layers = int(round((self_layers + other_layers) / 2, 0))

        if rand_rand() < 0.05:  # mutation to number of layers
            if rand_rand() < 0.5 and new_layers > 1:
                new_layers -= 1
            else:
                new_layers += 1

        new_generation = max(self.generation, other.generation) + 1

        def mutate_multiplier(mutability) -> float:
            return 1 + mutability * (rand_rand() - 0.5) / 25

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

        def create_new_coefficients(new_modifiers: dict, modifier_list: list, layer_name: str, coefficients) -> None:
            '''Modifies new_modifiers in place'''
            new_modifiers[layer_name][param] = {}
            len_modifier_list = len(modifier_list)
            new_mods_layername_param = new_modifiers[layer_name][param]  # local variable for speed
            for coef in coefficients:
                new_coef = sum(mods[layer_name][param][coef] for mods in modifier_list) / len_modifier_list
                if coef == 'X':
                    new_coef = round(new_coef * mutate_multiplier(new_mutability), 0)
                    if self.no_negative_exponents and new_coef < 0:
                        new_coef = 0
                else:
                    new_coef *= mutate_multiplier(new_mutability)
                new_mods_layername_param[coef] = new_coef


        possible_parameters = get_possible_parameters(self.full_parameter_example, self.target_parameter)
        coefficients = ['C', 'B', 'Z', 'X']
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
                            new_N *= mutate_multiplier(new_mutability)
                            new_modifiers[layer_name]['N'] = new_N
                        elif param in self_modifiers[layer_name]:
                            new_N = self_modifiers[layer_name][param] * mutate_multiplier(new_mutability)
                            new_modifiers[layer_name]['N'] = new_N
                        elif param in other_modifiers[layer_name]:
                            new_N = other_modifiers[layer_name][param] * mutate_multiplier(new_mutability)
                            new_modifiers[layer_name]['N'] = new_N
                    elif layer_name in self_modifiers:
                        new_N = self_modifiers[layer_name][param] * mutate_multiplier(new_mutability)
                        new_modifiers[layer_name]['N'] = new_N
                    elif layer_name in other_modifiers:
                        try:
                            new_N = other_modifiers[layer_name][param]
                        except KeyError:
                            breakpoint()
                        new_N *= mutate_multiplier(new_mutability)
                        new_modifiers[layer_name]['N'] = new_N

                else:  # param is one of ['T', 'B', 'C', 'X', 'Z']
                    force_keep_param = False if param != 'T' or layer == 1 else True
                    if layer_name in self_modifiers and layer_name in other_modifiers:
                        if param != 'T' or layer > 1:
                            if param in self_modifiers[layer_name] and param in other_modifiers[layer_name]:
                                create_new_coefficients(new_modifiers, [self_modifiers, other_modifiers], layer_name, coefficients)

                        elif param in self_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [self_modifiers], layer_name, coefficients)

                        elif param in other_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [other_modifiers], layer_name, coefficients)

                    elif layer_name in self_modifiers:
                        if param in self_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [self_modifiers], layer_name, coefficients)

                    elif layer_name in other_modifiers:
                        if param in other_modifiers[layer_name] and (rand_rand() < 0.5 or force_keep_param):
                            create_new_coefficients(new_modifiers, [other_modifiers], layer_name, coefficients)


        # Chance to add or remove parameter modifiers
        remove_modifiers = []
        add_modifiers = []
        for layer in range(1, new_layers + 1):
            layer_name = f'LAYER_{layer}'
            for param, values in new_modifiers[layer_name].items():
                if rand_rand() < 0.01 * new_mutability and param != 'T':
                    remove_modifiers.append((layer_name, param))
            for param in possible_parameters:
                if param not in new_modifiers[layer_name]:
                    if rand_rand() < 0.01 * new_mutability or (layer > 1 and param == 'T'):
                        add_modifiers.append((layer_name, param))

        for remove_tup in remove_modifiers:
            if remove_tup[1] != 'N':
                del new_modifiers[remove_tup[0]][remove_tup[1]]
            else:
                new_modifiers[remove_tup[0]][remove_tup[1]] = 0

        for add_tup in add_modifiers:
            if add_tup[1] == 'N':
                new_modifiers[add_tup[0]]['N'] = 0 if rand_rand() < 0.2 else random.gauss(0, new_mutability)
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


    def output_python_regression_module(self, output_filename: str='regression_function.py', standardizer=None):
        '''Create a Python module/file with a regression function represented by this EvogressionCreature'''
        output_str = self.output_regression_func_as_python_module_string(standardizer=standardizer)
        with open(output_filename, 'w') as f:
            f.write(output_str)
        print(f'EvogressionCreature modifiers outputted as regression function Python module!')


    def output_regression_func_as_python_module_string(self, standardizer=None) -> str:
        '''Create a string which creates a Python module with callable regression function.'''
        modifiers = self.simplify_modifiers(self.modifiers)
        used_parameters = {param for layer, layer_dict in modifiers.items() for param, coefficients in layer_dict.items()}

        s = "'''\nPython regression function module generated by Evogression.\n'''\n\n"
        s += "def regression_func(parameters: dict) -> float:\n"
        s += "    '''Generated by an Evogression creature'''\n\n"

        if len(used_parameters) > 0:
            if standardizer:
                s += "    # Standardize input data\n"
                s += "    standardized_data = {}\n"
                s += "    for param, value in parameters.items():\n"
                for param, modifier_dict in standardizer.data_modifiers.items():
                    if param != self.target_parameter and param in used_parameters:
                        s += f"        if param == '{param}':\n"
                        s += f"            standardized_data['{param}'] = (value - {round(modifier_dict['mean'], 6)}) / {round(modifier_dict['stdev'], 6)}\n"
                if '()' in s[-10:]:
                    s += f"        pass\n"
                s += f"    parameters = standardized_data\n\n"
            else:
                s += f"    # No standardizer used for regression\n\n"
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

        if standardizer:
            s += "    # Unstandardize result\n"
            s += f"    T = T * {round(standardizer.data_modifiers[self.target_parameter]['stdev'], 6)} + {round(standardizer.data_modifiers[self.target_parameter]['mean'], 6)}\n\n"

        s += "    return T\n"
        return s
