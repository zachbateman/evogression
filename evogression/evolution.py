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
                 max_layers: int=3,
                 max_cpu: int=max(os.cpu_count()-1, 1),  # by default use all but one core
                 optimize: bool=True,
                 **kwargs) -> None:

        self.target_parameter = target_parameter

        if isinstance(all_data, DataFrame):
            all_data = all_data.to_dict('records')

        self.param_medians = data.calc_param_medians(all_data, target_parameter)
        self.all_data = data.fill_none_with_median(all_data, target_parameter, self.param_medians)
        self.full_parameter_example = self.all_data[0]

        data.data_checks(self.all_data)

        self.num_creatures = int(num_creatures)
        self.num_cycles = int(num_cycles)
        self.max_layers = int(max_layers)

        os.environ['RAYON_NUM_THREADS'] = str(max_cpu)
        self.model = rust_evogression.run_evolution(target_parameter, self.all_data, num_creatures, num_cycles, max_layers, optimize)

        self.parameter_usefulness_count: dict = defaultdict(int)
        for creature in self.model.best_creatures:
            for param in creature.used_parameters():
                self.parameter_usefulness_count[param] += 1


    @property
    def best_creature(self):
        return self.model.best_creature

    @property
    def best_error(self):
        return self.model.best_error()


    def output_best_regression(self, output_filename='regression_function', directory: str='.', add_error_value=False) -> None:
        '''
        Save this the regression equation/function this evolution has found
        to be the best into a new Python module so that the function itself
        can be imported and used in other code.
        '''
        name_ext = f'___{round(self.model.best_error(), 4)}' if add_error_value else ''

        if directory != '.' and not os.path.exists(directory):
            os.mkdir(directory)

        if output_filename[-3:] == '.py':  # adding .py later; removing to easily check for periods
            output_filename = output_filename[:-3]

        output_filename = output_filename.replace('.', '_') # period in filename not valid
        output_filename = os.path.join(directory, output_filename + name_ext.replace('.', '_') + '.py')

        output_str = self.model.python_regression_module_string()
        with open(output_filename, 'w') as f:
            f.write(output_str)
        print('Evogression model saved as a Python module.')


    def save(self, filename='evolution_model') -> None:
        '''
        Save this Evolution Python object/model in a pickle file.
        Also removes non-essential data to reduce file size.
        '''
        # Clearing or shrinking unneeded attributes provides a smaller file size.
        self.all_data = None

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
        param_medians = self.param_medians  # local variable for speed
        param_example = self.full_parameter_example  # local variable for speed

        if prediction_key == '':
            prediction_key = f'{self.target_parameter}_PREDICTED'

        is_dataframe = True if isinstance(data, DataFrame) else False
        if is_dataframe:
            data = data.to_dict('records')  # will get processed as list

        if isinstance(data, list):  # DataFrames also get processed here (previously converted to a list)
            for d in data:
                for param, val in d.items():
                    if not val:  # make any None values the previously calculated median from the training data
                        d[param] = param_medians.get(param, 0.0)
                # Now remove any keys in data not in training data as will not be in regression and can cause issues.  Also don't want string values.
                clean = {k: v for k, v in d.items() if k in param_example and not isinstance(v, str)}
                d[prediction_key] = self.model.predict_point(clean)

            if is_dataframe:
                data = DataFrame(data)
            return data

        elif isinstance(data, dict):
            # make any None values the previously calculated median from the training data
            for param, val in data.items():
                if not val:
                    data[param] = param_medians.get(param, 0.0)
            # Now remove any keys in data not in training data as will not be in regression and can cause issues.  Also don't want string values.
            data = {k: v for k, v in data.items() if k in param_example and not isinstance(v, str)}
            data[prediction_key] = self.model.predict_point(data)
            return data

        else:
            print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')
