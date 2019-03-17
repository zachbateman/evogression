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
        modifiers = {}
        for layer in range(1, self.layers + 1):
            modifiers[f'LAYER_{layer}'] = {}
            for param in self.full_parameter_example.keys():
                # resist using parameters if many of them
                if random.random() < 1 / len(self.full_parameter_example) and param != self.target_parameter:
                    C, B, Z, X = self.generate_parameter_coefficients()
                    modifiers[f'LAYER_{layer}'][param] = {'C': C, 'B': B, 'Z': Z, 'X': X}
            if layer > 1:
                C, B, Z, X = self.generate_parameter_coefficients()
                modifiers[f'LAYER_{layer}']['T'] = {'C': C, 'B': B, 'Z': Z, 'X': X}
            modifiers[f'LAYER_{layer}']['N'] = 0 if random.random() < 0.2 else random.gauss(0, self.mutability)

        return modifiers

    def generate_parameter_coefficients(self):
        C = 1 if random.random() < 0.8 else random.gauss(1, self.mutability)
        B = 1 if random.random() < 0.4 else random.gauss(1, 2 * self.mutability)
        Z = 0 if random.random() < 0.6 else random.gauss(0, 3 * self.mutability)
        X = 1 if random.random() < 0.9 else random.gauss(1, 0.1 * self.mutability)
        return C, B, Z, X


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
        cost = 5 * self.layers  # cost will be AT LEAST 5
        cost += sum(len(layer_dict) for layer_dict in self.modifiers.values())
        return cost

    def __add__(self, other):
        '''
        Using the __add__ ('+') operator to mate the creatures and generate offspring.
        Offspring will have a combination of modifiers from both parents that
        also includes some mutation.
        '''
        combined_hunger = self.hunger + other.hunger
        chance_of_mating = (combined_hunger - 25) / 100

        if random.random() < (1 - chance_of_mating):
            return None

        # Generate new number of layers
        if self.layers == other.layers:
            new_layers = int(self.layers)
        elif abs(self.layers - other.layers) < 1:
            new_layers = int(self.layers) if random.random() < 0.5 else int(other.layers)

        if random.random() < 0.05:  # mutation to number of layers
            if random.random() < 0.5 and new_layers > 1:
                new_layers -= 1
            else:
                new_layers += 1

        new_generation = max(self.generation, other.generation) + 1

        def mutate_multiplier(mutability) -> float:
            return 1 + mutability * (random.random() - 0.5) / 100

        # Generate new mutability
        new_mutability = (self.mutability + other.mutability) / 2
        new_mutability *= mutate_multiplier(new_mutability)

        # Generate new modifier layer(s) based on self and other
        possible_parameters = ['N', 'T'].extend(sorted(key for key in self.full_parameter_example if key != self.target_parameter))
        coefficients = ['C', 'B', 'Z', 'X']
        new_modifiers = {f'LAYER_{layer}': {} for layer in range(1, new_layers)}
        for layer in range(1, new_layers + 1):
            layer_name = f'LAYER_{layer}'
            for param in possible_parameters:
                if param == 'N':
                    new_N = (self.modifiers[layer_name] + other.modifiers[layer_name]) / 2
                    new_N *= mutate_multiplier(new_mutability)
                    new_modifiers[layer_name]['N'] = new_N
                else:  # param is one of ['T', 'B', 'C', 'X', 'Z']
                    if param in self.modifiers[layer_name] and param in other.modifiers[layer_name]:
                        new_modifiers[layer_name][param] = {}
                        for coef in coefficients:
                            new_coef = (self.modifiers[layer_name][param][coef] + other.modifiers[layer_name][param][coef]) / 2
                            new_coef *= mutate_multiplier(new_mutability)
                            new_modifiers[layer_name][param][coef] = new_coef

                    elif param in self.modifiers[layer_name] and random.random() < 0.5:
                        new_modifiers[layer_name][param] = {}
                        for coef in coefficients:
                            new_coef = self.modifiers[layer_name][param][coef]
                            new_coef *= mutate_multiplier(new_mutability)
                            new_modifiers[layer_name][param][coef] = new_coef

                    elif param in other.modifiers[layer_name] and random.random() < 0.5:
                        new_modifiers[layer_name][param] = {}
                        for coef in coefficients:
                            new_coef = other.modifiers[layer_name][param][coef]
                            new_coef *= mutate_multiplier(new_mutability)
                            new_modifiers[layer_name][param][coef] = new_coef

        # Chance to add or remove parameter modifiers
        remove_modifiers = []
        add_modifiers = []
        for layer in range(1, new_layers + 1):
            for param, values in new_modifiers[layer_name].items():
                if random.random() < 0.01 * new_mutability:
                    remove_modifiers.append((f'LAYER_{layer}', param))
            for param in possible_parameters:
                if param not in new_modifiers[layer_name]:
                    if random.random() < 0.01 * new_mutability:
                        add_modifiers.append((f'LAYER_{layer}', param))

        for remove_tup in remove_modifiers:
            del new_modifiers[remove_tup[0]][remove_tup[1]]

        for add_tup in add_modifiers:
            if add_tup[1] == 'N':
                new_modifiers[add_tup[0]]['N'] = 0 if random.random() < 0.2 else random.gauss(0, new_mutability)
            else:
                C, B, Z, X = self.generate_parameter_coefficients()
                new_modifiers[add_tup[0]][add_tup[1]] = {'C': C, 'B': B, 'Z': Z, 'X': X}

        return EvogressionCreature(self.target_parameter, layers=new_layers, generation=new_generation, full_parameter_example=self.full_parameter_example, modifiers=new_modifiers)







    def __copy__(self):
        return EvogressionCreature(self.target_parameter, layers=self.layers, generation=self.generation, modifiers=self.modifiers)

    def __repr__(self) -> str:
        return 'EvogressionCreature'

    def get_regression_func(self):
        return self.modifiers
