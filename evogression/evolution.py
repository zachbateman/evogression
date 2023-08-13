'''
Module containing evolution algorithm for regression.
'''
from collections import defaultdict
import os
import pickle
from pandas import DataFrame
from . import data as data_funcs
from . import rust_evogression


class Evolution:
    '''
    EVOLUTION ALOGORITHM

    Evolves creatures by killing off the worst performers in
    each cycle and then randomly generating many new creatures.

    Input "all_data must have all numeric values (or None).
    '''
    def __init__(self,
                 target: str,
                 all_data: list[dict[str, float]] | DataFrame,
                 creatures: int=10000,
                 cycles: int=10,
                 max_layers: int=3,
                 max_cpu: int=max(os.cpu_count()-1, 1),  # by default use all but one core
                 optimize: bool=True,
                 ) -> None:
        self.target, self.creatures, self.cycles, self.max_layers = target, int(creatures), int(cycles), int(max_layers)

        if isinstance(all_data, DataFrame):
            all_data = all_data.to_dict('records')
        all_data = data_funcs.remove_blank_targets(all_data, target)
        self.param_medians = data_funcs.calc_param_medians(all_data, target)
        self.all_data = data_funcs.fill_none_with_median(all_data, target, self.param_medians)
        self.full_parameter_example = self.all_data[0]
        data_funcs.data_checks(self.all_data)

        os.environ['RAYON_NUM_THREADS'] = str(max_cpu)
        self.model = rust_evogression.run_evolution(target, self.all_data, creatures, cycles, max_layers, optimize)
        self.best_creature = self.model.best_creature
        self.best_error = self.model.best_error()

        self.parameter_usefulness_count: dict = defaultdict(int)
        for creature in self.model.best_creatures:
            for param in creature.used_parameters():
                self.parameter_usefulness_count[param] += 1


    def output_regression(self, output_filename='regression_function', directory: str='.', add_error_value=False) -> None:
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


    def predict(self, data: dict[str, float] | list[dict[str, float]] | DataFrame, prediction_key: str='', noprint: bool=True) -> dict[str, float] | list[dict[str, float]] | DataFrame:
        '''
        Add best_creature predictions to data arg as f'{target}_PREDICTED' new key (or as prediction_key kwarg).
        Return unstandardized dict or list of dicts or DataFrame depending on provided arg.
        '''
        target = self.target  # local variable for speed
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
            case DataFrame():  # process as a list
                data = DataFrame(self.predict(data.to_dict('records'), prediction_key=prediction_key, noprint=noprint))
            case list():
                for d, clean_row in zip(data, generate_clean_data(data)):
                    d[prediction_key] = self.model.predict_point(clean_row)
            case dict():
                data[prediction_key] = self.model.predict_point(generate_clean_data([data])[0])
            case _:
                print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')
        return data


    def save(self, filename: str="model.evo") -> None:
        self._model_serialized = self.model.to_json()
        self.model = None  # Delete actual Rust object as can't pickle it

        self._best_creature_serialized = self.best_creature.to_json()
        self.best_creature = None  # Delete actual Rust object as can't pickle it

        self.all_data = None  # Shrink object size - full data is not needed once initialized

        cleaned_name = filename + '.evo' if '.' not in filename else filename

        with open(cleaned_name, 'wb') as file:
            pickle.dump(self, file)


def load(filename: str="model.evo") -> Evolution:
    cleaned_name = filename + '.evo' if '.' not in filename else filename

    try:
        with open(cleaned_name, 'rb') as file:
            evo = pickle.load(file)
        evo.model = rust_evogression.load_evolution_from_json(evo._model_serialized)
        evo.best_creature = rust_evogression.load_creature_from_json(evo._best_creature_serialized)
        return evo
    except FileNotFoundError:
        print(f'Saved model not found.  Please check that it is spelled correctly: "{filename}"')
