'''
Module containing evolution algorithms for regression.
'''
from typing import List, Dict, Union
import statistics
import copy
import random
import math
import tqdm
import warnings
from collections import defaultdict
import easy_multip
import pickle
from pandas import DataFrame
# from .creatures import EvogressionCreature
from .standardize import Standardizer
from . import data
# try:
#     from .calc_error_sum import calc_error_sum
# except ImportError:
#     print('\nUnable to import Cython calc_error_sum module!')
#     print('If trying to install/run on a Windows computer, you may need to a C compiler.')
#     print('See: https://wiki.python.org/moin/WindowsCompilers')
#     print('  -> (If running Windows 7, try using Python 3.7 instead of 3.8+)\n')

from . import rust_evogression


# function must be module-level so that it is pickleable for multiprocessing
# making it a global variable that is a function set in evolution __init__ to be module-level but still customizable
find_best_creature_multip = None


class Evolution():
    '''
    EVOLUTION ALOGORITHM

    Evolves creatures by killing off the worst performers in
    each cycle and then randomly generating many new creatures.

    Input data must have all numeric values (or None).
    '''
    def __init__(self,
                 target_parameter: str,
                 all_data: Union[List[Dict[str, float]], DataFrame],
                 num_creatures: int=10000,
                 num_cycles: int=10,
                 force_num_layers: int=0,
                 max_layers: int=10,
                 num_cpu: int=1,
                 verbose: bool=True,
                 optimize = True,
                 **kwargs) -> None:

        self.target_parameter = target_parameter
        # self.verbose = verbose

        if isinstance(all_data, DataFrame):
            all_data = all_data.to_dict('records')

        data.data_checks(all_data)

        self.param_medians = data.calc_param_medians(all_data, target_parameter)
        self.all_data = data.fill_none_with_median(all_data, target_parameter, self.param_medians)

        self.num_creatures = int(num_creatures)
        self.num_cycles = int(num_cycles)
        # self.force_num_layers = int(force_num_layers)
        self.max_layers = int(max_layers)
        # self.num_cpu = num_cpu if num_cpu >= 1 else 1
        # global find_best_creature_multip
        # find_best_creature_multip = easy_multip.decorators.use_multip(find_best_creature, num_cpu=self.num_cpu)

        model = rust_evogression.run_evolution(target_parameter, self.all_data, num_creatures, num_cycles, max_layers)
        self.model = model
        
        self.parameter_usefulness_count: dict = defaultdict(int)
        for creature in model.best_creatures:
            for param in creature.used_parameters():
                self.parameter_usefulness_count[param] += 1


    # def evolution_cycle(self) -> None:
    #     '''
    #     Run one cycle of evolution that introduces new random creatures,
    #     kills weak creatures, and mates the remaining ones.
    #     '''
    #     self.kill_weak_creatures()
    #     self.mutate_top_creatures()
    #     self.mate_creatures()

    #     # Add random new creatures each cycle to get back to target num creatures
    #     # Or... cut out extra creatures if have too many (small chance of happening)
    #     if len(self.creatures) < self.num_creatures:
    #         self.creatures.extend([EvogressionCreature(self.target_parameter, full_parameter_example=self.all_data[0], layers=self.force_num_layers, max_layers=self.max_layers)
    #                                        for _ in range(int(round(self.num_creatures - len(self.creatures), 0)))])
    #     elif len(self.creatures) > self.num_creatures:
    #         self.creatures = self.creatures[:self.num_creatures]
    #     random.shuffle(self.creatures)  # used to mix up new creatures in among multip


    # def record_best_creature(self, best_creature, error) -> bool:
    #     '''
    #     Saves a copy of the provided best creature (and its error)
    #     into self.best_creatures list.

    #     Returns True/False depending on if the recorded creature is a new
    #     BEST creature (compared to previously recorded best creatures).

    #     Also record parameters used in regression equation (modifiers dict)
    #     to parameter_usefulness_count if a new best error/creatures.
    #     '''
    #     new_best_creature = False
    #     if error < self.best_error:
    #         new_best_creature = True
    #         # now count parameter usage if better than previous best creatures
    #         for param in best_creature.used_parameters():
    #             self.parameter_usefulness_count[param] += 1

    #     self.best_creatures.append([copy.deepcopy(best_creature), error])
    #     return new_best_creature


    # @property
    # def best_creature(self) -> EvogressionCreature:
    #     '''
    #     Return the best creature available in the self.best_creatures list.
    #     '''
    #     best_creature, best_error = None, 10 ** 150
    #     for creature_list in self.best_creatures:
    #         if creature_list[1] < best_error:
    #             best_error = creature_list[1]
    #             best_creature = creature_list[0]
    #     return best_creature


    # @property
    # def best_error(self) -> float:
    #     '''
    #     Return error associated with best creature available in self.best_creatures list.
    #     If no existing best_creatures, return default huge error.
    #     '''
    #     best_error = 10 ** 150
    #     for creature_list in self.best_creatures:
    #         if creature_list[1] < best_error:
    #             best_error = creature_list[1]
    #     return best_error


    # def mate_creatures(self) -> None:
    #     '''Mate creatures to generate new creatures'''
    #     rand_rand = random.random
    #     new_creatures = []
    #     append = new_creatures.append  # local for speed
    #     self_creatures = self.creatures  # local for speed
    #     for i in range(0, len(self.creatures), 2):
    #         if rand_rand() < 0.5:  # only a 50% chance of mating (cuts down on calcs and issues of too many creatures each cycle)
    #             creature_group = self_creatures[i: i + 2]
    #             try:
    #                 new_creature = creature_group[0] + creature_group[1]
    #                 if new_creature:
    #                     append(new_creature)
    #             except IndexError:  # occurs when at the end of self.creatures
    #                 pass
    #     self.creatures.extend(new_creatures)


    def output_best_regression(self, output_filename='regression_function', add_error_value=False) -> None:
        '''
        Save this the regression equation/function this evolution has found
        to be the best into a new Python module so that the function itself
        can be imported and used in other code.
        '''
        name_ext = f'___{round(self.best_error, 4)}' if add_error_value else ''

        if self.standardize:
            self.best_creature.output_python_regression_module(output_filename=output_filename, standardizer=self.standardizer, directory='regression_modules', name_ext=name_ext)
        else:
            self.best_creature.output_python_regression_module(output_filename=output_filename, directory='regression_modules', name_ext=name_ext)


    def save(self, filename='evolution_model') -> None:
        '''
        Save this Evolution Python object/model in a pickle file.
        Also removes non-essential data to reduce file size.
        '''
        # Clearing or shrinking these attributes provides a smaller file size.
        self.best_creatures[-10:]
        self.creatures = [self.best_creature]
        self.all_data = None
        self.standardized_all_data = None
        self.standardizer.all_data = None
        self.standardizer.standardized_data = None

        with open(filename if '.' in filename else filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, filename):
        '''
        Load an Evolution object from a saved, pickle file.
        '''
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except:
            with open(filename + '.pkl', 'rb') as f:
                return pickle.load(f)


    def predict(self, data: Union[Dict[str, float], List[Dict[str, float]], DataFrame], prediction_key: str='', standardized_data: bool=False):
        '''
        Add best_creature predictions to data arg as f'{target}_PREDICTED' new key.
        Return unstandardized dict or list of dicts or DataFrame depending on provided arg.
        '''
        target_param = self.target_parameter  # local variable for speed
        if prediction_key == '':
            prediction_key = f'{target_param}_PREDICTED'

        is_dataframe = True if type(data) == DataFrame else False
        if is_dataframe:
            data = data.to_dict('records')  # will get processed as list

        if isinstance(data, list):  # DataFrames also get processed here (previously converted to a list)
            # make any None values the previously calculated median from the training data
            for d in data:
                for param, val in d.items():
                    if not val:
                        d[param] = self.param_medians.get(param, 0.0)

            if not standardized_data and self.standardize:
                data = [self.standardizer.convert_parameter_dict_to_standardized(d) for d in data]
            parameter_example = self.best_creature.full_parameter_example
            for d in data:
                # errors result if leave in key:values not used in training (string split categories for example), so next line ensures minimum data is fed to .calc_target
                clean_d = {key: value for key, value in d.items() if key in parameter_example}
                d[prediction_key] = self.model.predict_point(clean_d)

            if self.standardize:
                unstandardized_data = []
                for d in data:
                    unstandardized_d = {}
                    for param, value in d.items():
                        unstandardized_d[param] = self.standardizer.unstandardize_value(target_param if param == prediction_key else param, value)
                    unstandardized_data.append(unstandardized_d)
            else:
                unstandardized_data = data
            if is_dataframe:
                unstandardized_data = DataFrame(unstandardized_data)
            return unstandardized_data

        elif isinstance(data, dict):
            # make any None values the previously calculated median from the training data
            for param, val in data.items():
                if not val:
                    data[param] = self.param_medians.get(param, 0.0)
            data[prediction_key] = self.model.predict_point(data)
            return data
        else:
            print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')
