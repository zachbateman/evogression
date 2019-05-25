'''
Module containing data standardization tools.
'''
import typing
import statistics
import copy


class Standardizer():

    def __init__(self,
                         all_data: typing.List[typing.Dict[str, float]]) -> None:
        self.all_data = all_data
        self.parameters = all_data[0].keys()
        # initially create self.standarized_data as copy of all_data
        self.standardized_data: typing.List[typing.Dict[str, float]] = copy.deepcopy(all_data)

        self.data_modifiers: typing.Dict[str, typing.Dict[str, float]] = {}
        self.data_modifiers = {p: {'mean': 0, 'stdev': 0} for p in self.parameters}

        for param in self.parameters:
            self.fully_standardize_parameter(param)


    def fully_standardize_parameter(self, param):
        '''
        Standardize one parameter/column of data
        '''
        values = [d[param] for d in self.all_data]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)

        self.data_modifiers[param]['mean'] = mean
        self.data_modifiers[param]['stdev'] = stdev

        new_values = [(v - mean) / stdev for v in values]
        for i in range(len(self.standardized_data)):
            self.standardized_data[i][param] = new_values[i]

    def convert_parameter_dict_to_standardized(self, param_dict: dict) -> dict:
        '''
        Convert a data point to the standardized equivalent dict.
        Used when testing new data points that have not been standardized as a whole.
        '''
        new_param_dict = {}
        for param, value in param_dict.items():
            new_param_dict[param] = (value - self.data_modifiers[param]['mean']) / self.data_modifiers[param]['stdev']
        return new_param_dict

    def get_standardized_data(self):
        return self.standardized_data

    def unstandardize_value(self, param, value) -> float:
        '''
        Convert a single value back into what it was/would have been without standardization
        '''
        data_mods_param = self.data_modifiers[param]  # local variable for speed
        return value * data_mods_param['stdev'] + data_mods_param['mean']
