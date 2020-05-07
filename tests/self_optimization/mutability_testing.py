
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '..\\..')
from test_data import surface_3d_data
import evogression
import easy_multip
import pandas
import statplot
import tqdm
from functools import lru_cache



def main1():
    data = [d for i, d in enumerate(surface_3d_data) if i % 10 == 0]
    standardizer = evogression.standardize.Standardizer(data)
    standardized_data = standardizer.get_standardized_data()
    unst_val = standardizer.unstandardize_value

    @lru_cache()
    def unst_val_actuals(param, val):
        return unst_val(param, val)

    creatures = easy_multip.map(get_evogressioncreature, range(50000))
    for cr in creatures:
        cr.error = 0

    for cr in tqdm.tqdm(creatures):
        for dp in standardized_data:
            cr.error += (unst_val('z', cr.calc_target(dp)) - unst_val_actuals('z', dp['z'])) ** 2

    errors = sorted(cr.error for cr in creatures)
    best_errors = set(errors[:int(round(0.6 * len(errors), 0))])

    creatures = [cr for cr in creatures if cr.error in best_errors]
    creatures = [cr for i, cr in enumerate(creatures) if i % 150 == 0]


    df = pandas.DataFrame()
    df['Mutability'] = [cr.mutability for cr in creatures]
    df['Error'] = [cr.error - min(errors) for cr in creatures]
    statplot.scatterplot(df, xvar='Mutability', yvar='Error', alpha=0.3)
    df['Mutability'] = [round(mut, 1) for mut in df['Mutability'].tolist()]
    statplot.distribution_plot(df, bin_col='Mutability', result_col='Error')


def get_evogressioncreature(iteration):
    return evogression.EvogressionCreature('z', full_parameter_example={'x': None, 'y': None, 'z': None},
                                           mutability = 0.5 * (iteration / 50000))


def main2():
    data = [d for i, d in enumerate(surface_3d_data) if i % 10 == 0]
    standardizer = evogression.standardize.Standardizer(data)
    standardized_data = standardizer.get_standardized_data()
    unst_val = standardizer.unstandardize_value

    mutability_best_errors = {}
    for mutability in tqdm.tqdm([0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]):
        creatures = [evogression.EvogressionCreature('z', full_parameter_example={'x': None, 'y': None, 'z': None}, mutability=mutability)
                                for _ in range(1500)]

        for cr in creatures:
            cr.error = 0

        for cr in creatures:
             for d in standardized_data:
                try:
                    cr.error += abs(unst_val('z', cr.calc_target(d)) - unst_val('z', d['z'])) ** 2
                except OverflowError:
                    cr.error = 10 ** 10

        errors = sorted(cr.error for cr in creatures)
        mutability_best_errors[mutability] = errors[:30]

    mutabilities = []
    errors = []
    for mut, error_lst in mutability_best_errors.items():
        mutabilities.extend([mut for _ in range(len(error_lst))])
        errors.extend(error_lst)

    df = pandas.DataFrame()
    df['Mutability'] = mutabilities
    df['Error'] = errors
    statplot.scatterplot(df, xvar='Mutability', yvar='Error', alpha=0.3)
    statplot.distribution_plot(df, bin_col='Mutability', result_col='Error')




if __name__ == '__main__':
    main1()
    main2()
