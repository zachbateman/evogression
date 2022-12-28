'''
Module containing evolution algorithms for regression.
'''
from typing import List, Dict, Union
from collections import defaultdict
import os
import pickle
from pandas import DataFrame
from . import data
from . import rust_evogression


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
                 max_cpu: int=max(os.cpu_count()-1, 1),  # by default use all but one core
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
        self.max_layers = int(max_layers)

        os.environ['RAYON_NUM_THREADS'] = str(max_cpu)
        self.model = rust_evogression.run_evolution(target_parameter, self.all_data, num_creatures, num_cycles, max_layers)

        self.parameter_usefulness_count: dict = defaultdict(int)
        for creature in self.model.best_creatures:
            for param in creature.used_parameters():
                self.parameter_usefulness_count[param] += 1


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
