
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '..\\..')
from test_data import surface_3d_data
import evogression



def main():
    evolution = evogression.Evolution('z', surface_3d_data, creatures=5000, cycles=5, use_multip=True)



if __name__ == '__main__':
    evolution = main()
