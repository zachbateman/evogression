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
        
        for col in df.columns:
            try:
                df = df[df[col] != '?']  # filter out rows without full data set.
            except TypeError:
                pass
            df[col] = df[col].map(lambda x: float(x))
            
        regression_data = df.to_dict('records')
        evolution = evogression.evolution.CreatureEvolution('benign2_or_malignant4', regression_data, target_num_creatures=5000, num_cycles=10)
        evolution.best_creatures[-1][0].output_python_regression_module()

        # x = [point_dict['x'] for point_dict in surface_3d_data]
        # y = [point_dict['y'] for point_dict in surface_3d_data]
        # z = [point_dict['z'] for point_dict in surface_3d_data]

        # standardized_3d_data = [evolution.standardizer.convert_parameter_dict_to_standardized(d) for d in surface_3d_data]
        # z_test = [evolution.standardizer.unstandardize_value('z', evolution.best_creature.calc_target(d)) for d in standardized_3d_data]

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter3D(x, y, z)
        # ax.scatter3D(x, y, z_test)

        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.title('Surface Regression - Evolution Test')

        # plt.show()



if __name__ == '__main__':
    unittest.main()
