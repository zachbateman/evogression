import math
import statistics


def data_checks(data_to_check: list[dict]) -> None:
    '''
    Check cleaned input data for potential issues.
    At the point when this is called, there should be no issues with
    the data to be used; data-cleaning methods are called earlier.

    If this method prints errors, we need to write more data-cleaning capabilities to handle those cases!
    '''
    acceptable_types = {'float', 'int', 'float64', 'int64'}
    issues = []
    def check_data(data, data_name):
        for i, d in enumerate(data):
            for key, val in d.items():
                # val > & < checks are way of checking for nan without needing to require numpy import
                if type(val).__name__ not in acceptable_types or not (val >= 0 or val <= 0):
                    issues.append((data_name, i, key, val))

    check_data(data_to_check, 'all_data')
    for issue in issues:
        data_name, i, key, val = issue
        print(f'\nERROR!  NAN values detected in {data_name}!')
        print(f'Index: {i}  key: {key}  value: {val}  type: {type(val).__name__}')


def calc_param_medians(data: list[dict], target: str) -> dict:
     # Remove any data points that have None for the target/result parameter
    data = [d for d in data if d[target] is not None and not math.isnan(d[target])]

    is_nan = math.isnan  # local for speed
    param_medians = {}
    for param in data[0].keys():
        if param != target:
            values = [val for d in data if (val := d[param]) is not None and not is_nan(val)]
            param_medians[param] = statistics.median(values) if values else 0.0  # if values is empty, set all to zero
    return param_medians


def fill_none_with_median(data: list[dict], target: str, param_medians: dict) -> list[dict]:
        '''
        Find median value of each input parameter and
        then replace any None values with this median.
        '''
        # Remove any data points that have None for the target/result parameter
        data = [d for d in data if d[target] is not None and not math.isnan(d[target])]

        # param_medians = {}  # used for default parameter values if None is provided in .predict
        is_nan = math.isnan  # local for speed
        parameters_adjusted = set()
        for param in (key for key in data[0].keys() if key != target):
            median = param_medians[param]
            for d in data:
                if (val := d[param]) is None or is_nan(val):
                    parameters_adjusted.add(param)
                    d[param] = median

        if len(parameters_adjusted) >= 1:
            print('Data None values filled with median for the following parameters:')
            for param in sorted(parameters_adjusted):
                print(f'  {param}')

        return data


class InputDataFormatError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return '\n' + str(self.message) if self.message else '\n'
