import math
import statistics


def data_checks(data_to_check: list[dict]) -> None:
    '''
    Check cleaned input data for potential issues.
    At the point when this is called, there should be no issues with
    the data to be used; data-cleaning methods are called earlier.
    '''
    is_nan = math.isnan  # local for speed
    acceptable_types = {'float', 'int', 'float64', 'int64'}
    for i, d in enumerate(data_to_check):
        for key, val in d.items():
            if type(val).__name__ not in acceptable_types or is_nan(val):
                raise InputDataFormatError(f'Index: {i}  key: {key}  value: {val}  type: {type(val).__name__}')


def calc_param_medians(data: list[dict], target: str) -> dict:
    data = remove_blank_targets(data, target)
    is_nan = math.isnan  # local for speed
    param_medians = {}
    for param in data[0].keys():
        if param != target:
            if any((isinstance(d[param], str) for d in data)):
                raise InputDataFormatError(f'  - String/text values detected in column "{param}".\n  - Please remove these or convert to a number.')
            values = [val for d in data if (val := d[param]) is not None and not is_nan(val)]
            param_medians[param] = statistics.median(values) if values else 0.0  # if values is empty, set all to zero
    return param_medians


def remove_blank_targets(data: list[dict], target: str) -> list[dict]:
    '''Remove any data points that have None for the target/result parameter'''
    is_nan = math.isnan  # local for speed
    return [d for d in data if (val := d[target]) is not None and not is_nan(val)]


def fill_none_with_median(data: list[dict], target: str, param_medians: dict, noprint: bool=False) -> list[dict]:
    '''
    Replace any None values of input data with their median value.
    '''
    is_nan = math.isnan  # local for speed
    parameters_adjusted = set()
    for param in (key for key in data[0].keys() if key != target):
        median = param_medians.get(param, 0.0)
        for d in data:
            if (val := d[param]) is None or is_nan(val):
                parameters_adjusted.add(param)
                d[param] = median
    if not noprint and len(parameters_adjusted) >= 1:
        print('None values filled with median for the following parameters:\n  ' + '\n  '.join(sorted(parameters_adjusted)))
    return data


class InputDataFormatError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return '\n' + str(self.message) if self.message else '\n'
