'''
Module containing evolution algorithms for regression.
'''
from collections import defaultdict
import os
import pickle
from pandas import DataFrame
from . import data as data_funcs
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
                 all_data: list[dict[str, float]] | DataFrame,
                 num_creatures: int=10000,
                 num_cycles: int=10,
                 max_layers: int=3,
                 max_cpu: int=max(os.cpu_count()-1, 1),  # by default use all but one core
                 optimize: bool=True,
                 ) -> None:

        self.target_parameter = target_parameter

        if isinstance(all_data, DataFrame):
            all_data = all_data.to_dict('records')

        all_data = data_funcs.remove_blank_targets(all_data, target_parameter)
        self.param_medians = data_funcs.calc_param_medians(all_data, target_parameter)
        self.all_data = data_funcs.fill_none_with_median(all_data, target_parameter, self.param_medians)
        self.full_parameter_example = self.all_data[0]

        data_funcs.data_checks(self.all_data)

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


    def predict(self, data: dict[str, float] | list[dict[str, float]] | DataFrame, prediction_key: str='', noprint: bool=True):
        '''
        Add best_creature predictions to data arg as f'{target}_PREDICTED' new key.
        Return unstandardized dict or list of dicts or DataFrame depending on provided arg.
        '''
        target = self.target_parameter  # local variable for speed
        param_example = self.full_parameter_example  # local variable for speed
        prediction_key = prediction_key if prediction_key != '' else f'{target}_PREDICTED'

        def generate_clean_data(data: list[dict]) -> list[dict]:
            clean_data = []
            for d in data:
                # Remove any keys in data not in training data as will not be in regression and could cause issues.  Also don't want string values.
                clean = {k: v for k, v in d.items() if k in param_example and not isinstance(v, str)}
                # Add in any missing keys with the median value for that parameter
                for key in param_example:
                    if key not in clean and key != target:
                        clean[key] = None
                clean_data.append(clean)
            return data_funcs.fill_none_with_median(clean_data, target, self.param_medians, noprint=noprint)

        match data:
            case DataFrame():
                data = data.to_dict('records')  # will get processed as list
                for d, clean_row in zip(data, generate_clean_data(data)):
                    d[prediction_key] = self.model.predict_point(clean_row)
                data = DataFrame(data)  # convert back into a DataFrame
            case list():
                for d, clean_row in zip(data, generate_clean_data(data)):
                    d[prediction_key] = self.model.predict_point(clean_row)
            case dict():
                data[prediction_key] = self.model.predict_point(generate_clean_data([data])[0])
            case _:
                print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')
        return data
