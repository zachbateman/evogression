'''
Module containing high-level evogression functionality to fit and summarize regressions.
'''
import random
from math import log
from collections import defaultdict
from pandas import DataFrame
from .evolution import Evolution


class EvoGroup:
    def __init__(self, models: list[Evolution]):
        self.models: list[Evolution] = models

    def __iter__(self):
        return (model for model in self.models)

    def output_regression(self, directory: str='regression_modules') -> None:
        '''Output each Evolution's regression as a new Python module.'''
        for model in self.models:
            model.output_regression(directory=directory, add_error_value=True)

    @property
    def parameter_usage(self) -> dict[str, int]:
        '''Combine each Evolution's .parameter_usefulness_count dicts to see which attributes are important.'''
        combined_parameter_usefulness = defaultdict(int)
        for model in self.models:
            for param, count in model.parameter_usefulness_count.items():
                combined_parameter_usefulness[param] += count
        return combined_parameter_usefulness

    def output_param_usage(self, filename: str='ParameterUsage.xlsx') -> None:
        '''Save an Excel file with Parameter Usage data.'''
        usage = DataFrame.from_dict(self.parameter_usage, orient='index').reset_index().rename(columns={'index': 'PARAMETER', 0: 'USAGE'})
        usage.sort_values('USAGE', ascending=False, inplace=True)
        usage.to_excel(filename if '.' in filename else filename + '.xlsx', index=False, sheet_name='Parameter Usage')

    def predict(self, data: dict[str, float] | list[dict[str, float]] | DataFrame, prediction_key: str=''):
        '''Predict like Evogression.predict() but average this group's predictions.'''
        match data:
            case DataFrame():  # will get processed as a list
                return DataFrame(self.predict(data.to_dict('records'), prediction_key=prediction_key))
            case list():
                for row in data:
                    row[prediction_key] = sum(model.predict(row, prediction_key)[prediction_key] for model in self.models) / len(self.models)
            case dict():
                data[prediction_key] = sum(model.predict(data, prediction_key)[prediction_key] for model in self.models) / len(self.models)
            case _:
                print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')
        return data


def evolution_group(target: str, data: list[dict[str, float]] | DataFrame, creatures: int=10000, cycles: int=10,
                    max_layers: int=3, group_size: int=4, max_cpu: int=0, optimize=True, **kwargs) -> EvoGroup:
    '''Generate an EvoGroup containing multiple fully initialized Evolution objects.  Any Evolution kwargs may be provided.'''
    group = [Evolution(target, data, creatures=creatures, cycles=cycles, max_layers=max_layers,  max_cpu=max_cpu, optimize=optimize, **kwargs)
                for _ in range(group_size)]
    return EvoGroup(group)


def generate_robust_param_usage_file(target: str, data: list[dict[str, float]] | DataFrame, num_models: int=100, creatures: int=5000,
                                     cycles: int=3, filename: str='RobustParameterUsage.xlsx') -> None:
    '''
    Generate many models using subsets of possible input data columns so as to
    not overweight usage of the few best columns.
    Output the parameter usage/predictability of the data columns to Excel.
    '''
    if isinstance(data, list):
        data = DataFrame(data)

    columns = list(data.columns)
    num_col_per_sample = len(columns) if len(columns) <= 4 else int(7 * log(len(columns)) - 8) + 1

    models = []
    for _ in range(num_models):
        col_subset = set(random.sample(columns, num_col_per_sample) + [target])  # need to ensure prediction column in the data
        data_subset = data[list(col_subset)]  # need to pass in list instead of a set
        models.append(Evolution(target, data_subset, creatures=creatures, cycles=cycles, optimize=False))
    EvoGroup(models).output_param_usage(filename=filename)


def parameter_pruned_evolution_group(target: str, data: list[dict[str, float]] | DataFrame, max_parameters: int=10, creatures: int=10000,
                                     cycles: int=10, group_size: int=4) -> EvoGroup:
    '''
    Generate successive groups of Evolution objects and prune least-used
    parameters from the input data each round until only the most useful parameters remain.
    Finally, run a full-blown evolution cycles with the remaining parameters
    for the saved regression modules.

    USE LARGE >>> cycles <<< TO INTEGRATE MORE STATISTICALLY VALID PARAMETER USAGE NUMBERS BEFORE REMOVING PARAMETERS
    '''
    if creatures < 1000 or cycles < 10:
        print('\n> ERROR!  parameter_pruned_evolution_group() can be unstable (infinite loop)\n  if creatures and/or cycles args are too small.')
        print('  Please use a minimum creatures of 1000 and a minimum ncycles of 10.\n  Higher values cycles are encouraged!\n')
        return

    def num_param_to_eliminate(num_extra_param: int) -> int:
        '''Determine how many worst-performing parameters to elimintate in a given round'''
        if num_extra_param > 50:
            return num_extra_param // 2
        elif num_extra_param > 10:
            return num_extra_param // 3
        elif num_extra_param > 3:
            return 2
        elif num_extra_param > 0:
            return 1
        else:
            return 0

    if isinstance(data, DataFrame):
        data = data.to_dict('records')

    num_parameters = len(data[0].keys()) - 1
    while num_parameters > max_parameters:
        group = evolution_group(target, data, int(creatures // 1.6), int(cycles // 1.6), group_size=group_size, optimize=False)

        current_parameter_usage = [(param, count) for param, count in group.parameter_usage.items()]
        random.shuffle(current_parameter_usage)  # so below filter ignores previous order for equally-ranked parameters
        ranked_parameters = sorted(current_parameter_usage, key=lambda tup: tup[1])
        print(ranked_parameters)

        dead_params = [t[0] for t in ranked_parameters[:num_param_to_eliminate(num_parameters - max_parameters)]]
        for data_point in data:
            for param in dead_params:
                del data_point[param]
        num_parameters = len(data[0].keys()) - 1

    final_group = evolution_group(target, data, creatures, cycles, group_size=group_size)
    print('parameter_pruned_evolution_group complete.  Final Parameter usage counts below:')
    for param, count in final_group.parameter_usage.items():
        print(f'  {count}: {param}')
    return final_group


def random_population(target: str, data: list[dict[str, float]], creatures: int=10000, cycles: int=10, group_size: int=4, **kwargs) -> EvoGroup:
    '''
    Generate a list of Evolution objects (same as evolution_group) but use randomly sampled data subsets for training.
    The goal is to generate a "Random Population" in a similar manner as a Random Forest concept.
    '''
    data_subset_size = int(len(data) * 1.5 // group_size)
    evolutions = []
    for _ in range(group_size):
        data_subset = random.choices(data, k=data_subset_size)
        evolutions.append(Evolution(target, data_subset, creatures=creatures, cycles=cycles, **kwargs))
    return EvoGroup(evolutions)


class Population:
    def __init__(self, target: str, data: list[dict[str, float]] | DataFrame, creatures=300, cycles: int=3, group_size: int=4,
                 split_parameter=None, category_or_continuous='category', bin_size=None, **kwargs):
        self.target = target

        if category_or_continuous not in ['category', 'continuous']:
            print('ERROR! Population category_or_continuous kwarg must be either "category" or "continuous"!')
            return

        if isinstance(data, DataFrame):
            data = data.to_dict('records')

        self.split_parameter = split_parameter
        self.category_or_continuous = category_or_continuous
        if split_parameter and category_or_continuous == 'category':
            categories = set(d[split_parameter] for d in data)
            data_sets = {cat: [{k: v for k, v in d.items() if k != split_parameter} for d in data if d[split_parameter] == cat] for cat in categories}
            self.evo_sets = {}
            for cat, data_subset in data_sets.items():
                self.evo_sets[cat] = evolution_group(target, data_subset, creatures=creatures, cycles=cycles, group_size=group_size, **kwargs)

        elif split_parameter and category_or_continuous == 'continuous':
            split_values = sorted(d[split_parameter] for d in data)
            if not bin_size:  # Use bin_size arg (or generate if not provided) to determine how to split out data into different bins of the split_parameter.
                bin_size = (max(split_values) - min(split_values)) / 5

            self.bins, self.evo_sets = [], {}
            lower = min(split_values)
            upper = lower + bin_size
            while lower < max(split_values):
                self.bins.append((lower, upper))
                cutoff_lower, cutoff_upper = lower - bin_size * 0.5, upper + bin_size * 0.5
                self.evo_sets[self.bins[-1]] = evolution_group(target, [d for d in data if cutoff_lower <= d[split_parameter] < cutoff_upper], creatures=creatures, cycles=cycles, group_size=group_size, **kwargs)
                lower, upper = upper, upper + bin_size


    def predict(self, data: dict[str, float] | list[dict[str, float]] | DataFrame, prediction_key: str='', smooth: bool=True):
        '''Generate predictions with same interface as Evolution.predict.'''
        if prediction_key == '':
            prediction_key = f'{self.target}_PREDICTED'

        if self.category_or_continuous == 'category':
            match data:
                case DataFrame():  # process as a list
                    data = DataFrame(self.predict(data.to_dict('records'), prediction_key=prediction_key))
                case list():
                    for d in data:
                        d[prediction_key] = self.evo_sets[d[self.split_parameter]].predict(d, prediction_key)[prediction_key]
                case dict():
                    data[prediction_key] = self.evo_sets[data[self.split_parameter]].predict(data, prediction_key)[prediction_key]
                case _:
                    print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')

        elif self.category_or_continuous == 'continuous':
            match data:
                case DataFrame():  # process as a list
                    data = DataFrame(self.predict(data.to_dict('records'), prediction_key=prediction_key))
                case list():
                    evos = self.evo_sets  # local for speed
                    split_p = self.split_parameter  # local for speed
                    if smooth:
                        bin_dist = lambda val, bin: abs(val - (bin[0]+bin[1]) / 2)
                        for row in data:
                            # Get closest 2 bins with their distance to current point
                            (bin1, b1_dist), (bin2, b2_dist) = sorted([(bin, bin_dist(row[split_p], bin)) for bin in self.bins], key=lambda tup: tup[1])[:2]
                            total_dist = b1_dist + b2_dist
                            # Generate weighted-average prediction by giving more weight to closer bin and less to farther one
                            row[prediction_key] = evos[bin1].predict(row, 'pred')['pred'] * (1 - b1_dist / total_dist) + evos[bin2].predict(row, 'pred')['pred'] * (1 - b2_dist / total_dist)
                    else:
                        for row in data:
                            bin_val = row[split_p]
                            for bin in self.bins:
                                if bin[0] <= bin_val <= bin[1]:
                                    row[prediction_key] = evos[bin].predict(row, prediction_key)[prediction_key]
                                    break
                case dict():  # process as a list
                    data = self.predict([data], prediction_key=prediction_key)[0]

        return data
