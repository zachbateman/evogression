'''
Module containing high-level evogression functionality to fit and summarize regressions.
'''
import random
from math import log
from collections import defaultdict
import pandas
from pandas import DataFrame
from .evolution import Evolution



def evolution_group(target_param: str, data: list[dict[str, float]] | DataFrame, num_creatures: int=10000, num_cycles: int=10,
                    max_layers: int=3, group_size: int=4, max_cpu: int=0, optimize=True, **kwargs) -> list[Evolution]:
    '''
    Generate a list of fully initialized Evolution objects.
    Any Evolution kwargs may be provided.
    '''
    return [Evolution(target_param, data, num_creatures=num_creatures, num_cycles=num_cycles, max_layers=max_layers,  max_cpu=max_cpu, optimize=optimize, **kwargs)
                for _ in range(group_size)]


def output_group_regression_funcs(group: list) -> None:
    '''
    Take list of Evolution objects and output their
    best regressions to a "regression_modules" subdir.
    '''
    for cr_ev in group:
        cr_ev.output_best_regression(directory='regression_modules', add_error_value=True)


def parameter_usage(group: list[Evolution]) -> dict[str, int]:
    '''
    Combine each Evolution's .parameter_usefulness_count dicts to see which attributes are important.
    '''
    combined_parameter_usefulness = defaultdict(int)
    for cr_ev in group:
        for param, count in cr_ev.parameter_usefulness_count.items():
            combined_parameter_usefulness[param] += count
    return combined_parameter_usefulness


def output_usage(group: list[Evolution], filename: str='ParameterUsage.xlsx') -> None:
    usage = parameter_usage(group)
    usage = pandas.DataFrame.from_dict(usage, orient='index')
    usage.reset_index(inplace=True)
    usage.columns = ['PARAMETER', 'USAGE']
    usage.sort_values('USAGE', ascending=False, inplace=True)
    usage.to_excel(filename if '.' in filename else filename + '.xlsx', index=False, sheet_name='Param Usage')


def generate_parameter_usage_file(data: DataFrame, column: str, num_models: int=100, num_creatures: int=5000,
                                  num_cycles: int=3,num_cpu: int=1, filename: str='ParameterUsage.xlsx') -> None:
    '''
    Generate many models using subsets of possible input data columns so as to
    not overweight usage of the few best columns.
    Output the parameter usage/predictability of the data columns to Excel.

    CURRENTLY only accepts a DataFrame for data arg.
    '''
    columns = list(data.columns)
    if len(columns) <= 4:
        num_col_per_sample = len(columns)
    else:
        num_col_per_sample = int(7 * log(len(columns)) - 8) + 1

    models = []
    for _ in range(num_models):
        col_subset = random.sample(columns, num_col_per_sample)
        if column not in col_subset:  # need prediction column in the data
            col_subset.append(column)
        data_subset = data[col_subset]
        model = Evolution(column, data_subset, num_creatures=num_creatures, num_cycles=num_cycles, num_cpu=num_cpu, optimize=False)
        model.clear_data()  # clear out references to data to shrink model object and limit memory growth
        models.append(model)
    output_usage(models, filename)


def parameter_pruned_evolution_group(target_param: str, data: list, max_parameters: int=10, num_creatures: int=10000,
                                     num_cycles: int=10, group_size: int=4) -> list[Evolution]:
    '''
    Generate successive groups of Evolution objects and prune least-used
    parameters from the input data each round until only the most useful parameters remain.
    Finally, run a full-blown evolution cycles with the remaining parameters
    for the saved regression modules.

    USE LARGE >>> num_cycles <<< TO INTEGRATE MORE STATISTICALLY VALID PARAMETER USAGE NUMBERS BEFORE REMOVING PARAMETERS
    '''
    if num_creatures < 1000 or num_cycles < 10:
        print('\n> ERROR!  parameter_pruned_evolution_group() can be unstable (infinite loop)\n  if num_creatures and/or num_cycles args are too small.')
        print('  Please use a minimum num_creatures of 1000 and a minimum num_cycles of 10.\n  Higher values num_cycles are encouraged!\n')
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

    num_parameters = len(data[0].keys()) - 1
    while num_parameters > max_parameters:
        group = evolution_group(target_param, data, int(num_creatures // 1.6), int(num_cycles // 1.6), group_size=group_size, optimize=False)

        current_parameter_usage = [(param, count) for param, count in parameter_usage(group).items()]
        random.shuffle(current_parameter_usage)  # so below filter ignores previous order for equally-ranked parameters
        ranked_parameters = sorted(current_parameter_usage, key=lambda tup: tup[1])
        print(ranked_parameters)

        dead_params = [t[0] for t in ranked_parameters[:num_param_to_eliminate(num_parameters - max_parameters)]]
        for data_point in data:
            for param in dead_params:
                del data_point[param]
        num_parameters = len(data[0].keys()) - 1

    final_group = evolution_group(target_param, data, num_creatures, num_cycles, group_size=group_size)
    print('parameter_pruned_evolution_group complete.  Final Parameter usage counts below:')
    for param, count in parameter_usage(final_group).items():
        print(f'  {count}: {param}')
    return final_group


def random_population(target_param: str, data: list[dict[str, float]], num_creatures: int=10000, num_cycles: int=10, group_size: int=4, **kwargs) -> list[Evolution]:
    '''
    Generate a list of Evolution objects (same as evolution_group) but use randomly sampled data subsets for training.
    The goal is to generate a "Random Population" in a similar manner as a Random Forest concept.
    '''
    data_subset_size = int(len(data) * 1.5 // group_size)
    evolutions = []
    for _ in range(group_size):
        data_subset = random.choices(data, k=data_subset_size)
        evolutions.append(Evolution(target_param, data_subset, num_creatures=num_creatures, num_cycles=num_cycles, **kwargs))
    return evolutions


class Population():
    def __init__(self, target_param: str, data: list[dict[str, float]], num_creatures=300, num_cycles: int=3, group_size: int=4,
                 split_parameter=None, category_or_continuous='category', bin_size=None, **kwargs):
        self.target_parameter = target_param

        if type(data) == DataFrame:
            data = data.to_dict('records')

        self.split_parameter = split_parameter
        self.category_or_continuous = category_or_continuous
        if split_parameter and category_or_continuous == 'category':
            categories = set(d[split_parameter] for d in data)
            data_sets = {cat: [{k: v for k, v in d.items() if k != split_parameter} for d in data if d[split_parameter] == cat] for cat in categories}
            self.evo_sets = {}
            for cat, data_subset in data_sets.items():
                self.evo_sets[cat] = [Evolution(target_param, data_subset, num_creatures=num_creatures, num_cycles=num_cycles, **kwargs) for _ in range(group_size)]

        elif split_parameter and category_or_continuous == 'continuous':
            # Use bin_size arg (or generate if not provided) to determine how to split out data into different bins of the split_parameter.
            split_values = sorted(d[split_parameter] for d in data)
            if not bin_size:
                bin_size = (max(split_values) - min(split_values)) / 5

            bins = []
            binned_data = {}
            lower = min(split_values)
            upper = lower + bin_size
            while lower < max(split_values):
                bins.append((lower, upper))
                cutoff_lower = lower - bin_size * 0.5
                cutoff_upper = upper + bin_size * 0.5
                binned_data[(lower, upper)] = [d for d in data if cutoff_lower <= d[split_parameter] < cutoff_upper]
                lower, upper = upper, upper + bin_size
            self.bins = bins

            self.evo_sets = {}
            for bin in bins:
                self.evo_sets[bin] = evolution_group(target_param, binned_data[bin], num_creatures=num_creatures, num_cycles=num_cycles, group_size=group_size, **kwargs)



    def predict(self, data: dict[str, float] | list[dict[str, float]] | DataFrame, prediction_key: str=''):
        '''
        Generate predictions with same interface as BaseEvolution.predict.
        '''
        if prediction_key == '':
            prediction_key = f'{self.target_parameter}_PREDICTED'

        is_dataframe = True if type(data) == DataFrame else False
        if is_dataframe:
            data = data.to_dict('records')  # will get processed as a list

        if self.category_or_continuous == 'category':
            if isinstance(data, list):  # dataframes also get processed here
                for d in data:
                    predictions = [evo.predict(d, 'pred')['pred'] for evo in self.evo_sets[d[self.split_parameter]]]
                    d[prediction_key] = sum(predictions) / len(predictions)
                return DataFrame(data) if is_dataframe else data
            elif isinstance(data, dict):
                predictions = [evo.predict(data, 'pred')['pred'] for evo in self.evo_sets[data[self.split_parameter]]]
                data[prediction_key] = sum(predictions) / len(predictions)
                return data
            else:
                print('Error!  "data" arg provided to .predict() must be a dict or list of dicts or DataFrame.')


        elif self.category_or_continuous == 'continuous':
            if isinstance(data, list):  # dataframes also get processed here
                data_points = data
                is_dict = False
            elif isinstance(data, dict):
                data_points = [data]
                is_dict = True

            bin_dist = lambda val, bin: abs(val - (bin[1]+bin[0]) / 2)
            for data_point in data_points:
                bin_distances = sorted([(bin, bin_dist(data_point[self.split_parameter], bin)) for bin in self.bins], key=lambda tup: tup[1])
                min_dist = bin_distances[0][1] if bin_distances[0][1] > 0 else bin_distances[1][1]
                bin_distances = [(t[0], t[1] + min_dist) for t in bin_distances]  # decrease distance sensitivity a hair (still include other weight is closest is almost exactly the value)
                bin_distances = bin_distances[:-2 * int(len(bin_distances) / 3)]  # remove furthest 2/3 of bins from consideration
                total_dists = sum(t[1] for t in bin_distances)

                # Now generate weighted-average prediction by giving more weight to closer bins and less to further ones
                total_prediction = 0.0
                for bin, dist in bin_distances:
                    predictions = sorted(evo.predict(data_point, 'pred', noprint=True)['pred'] for evo in self.evo_sets[bin])
                    bin_pred = sum(predictions[1:-1]) / (len(predictions) - 2)  # kick out max/min (outlyer) predictions
                    total_prediction += bin_pred * (dist / total_dists)

                data_point[prediction_key] = total_prediction

            if is_dict:
                return data_points[0]
            elif is_dataframe:
                return DataFrame(data_points)
            else:
                return data_points
