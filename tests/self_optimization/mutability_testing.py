
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



def main():
    data = surface_3d_data
    data = [d for i, d in enumerate(data) if i % 10 == 0]
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



if __name__ == '__main__':
    evolution = main()
