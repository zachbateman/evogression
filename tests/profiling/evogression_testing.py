
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '..\\..')
from test_data import parabola_data
import evogression



def main():
    # evolution = evogression.Evolution('y', parabola_data, num_creatures=10000, num_cycles=10, use_multip=False, initial_creature_creation_multip=False, optimize=3)
    evolution = evogression.Evolution('y', parabola_data, num_creatures=100000, num_cycles=50, use_multip=False, initial_creature_creation_multip=False, optimize=3)



if __name__ == '__main__':
    evolution = main()
