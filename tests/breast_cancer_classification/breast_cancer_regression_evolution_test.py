import unittest
# import sys
# sys.path.insert(1, '..')
# sys.path.insert(1, '..\\..')
import evogression
from pprint import pprint as pp
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

        evolution = evogression.Evolution('benign2_or_malignant4', df, target_num_creatures=10000, cycles=10)
        evolution.output_best_regression()

        output_data = evolution.predict(df)
        output_data.to_excel('BreastCancerPredictions_new.xlsx')



if __name__ == '__main__':
    unittest.main()
