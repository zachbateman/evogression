'''
Cython implementation of EvogressionCreature._generate_parameter_coefficients()
'''
import random


cpdef tuple generate_parameter_coefficients_calc():

    cpdef double C
    cpdef double B
    cpdef double Z
    cpdef double X

    rand_rand = random.random  # local for speed
    rand_tri = random.triangular  # local for speed
    C = 1 if rand_rand() < 0.4 else rand_tri(0, 2, 1)
    B = 1 if rand_rand() < 0.3 else rand_tri(0, 2, 1)
    Z = 0 if rand_rand() < 0.4 else rand_tri(-2, 2, 0)
    if rand_rand() < 0.5:
        C = -C
    if rand_rand() < 0.5:
        B = -B
    X = 1 if rand_rand() < 0.4 else random.choice([0, 2, 2, 2, 2, 2, 3, 3])
    return C, B, Z, X
