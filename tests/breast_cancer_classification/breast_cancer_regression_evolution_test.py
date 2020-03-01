import unittest
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '..\\..')
import evogression
from test_data import surface_3d_data
from pprint import pprint as pp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas


class TestBreastCancerDetectionRegression(unittest.TestCase):

    def test_breast_cancer_detection(self):
        df = pandas.read_csv('breast-cancer-wisconsin.data')
        df.drop('id_num', axis=1, inplace=True)
        df = df.replace('?', None)

        for col in df.columns:
            vals = df[col].tolist()
            if '?' in vals:
                print(vals)
            try:
                df[col] = df[col].map(lambda x: float(x) if x is not None else x)
            except:
                print(f'ERROR column: {col}')

        regression_data = df.to_dict('records')
        evolution = evogression.Evolution('benign2_or_malignant4', regression_data, target_num_creatures=5000, num_cycles=10)
        evolution.best_creature.output_python_regression_module()

        output_data = evolution.add_predictions_to_data(regression_data)
        output_df = pandas.DataFrame(output_data)
        output_df.to_excel('BreastCancerPredictions.xlsx')



if __name__ == '__main__':
    unittest.main()
