'''
Cython implementation of EvogressionCreature._generate_parameter_coefficients()
'''
import random


cpdef (double, double, double, int) generate_parameter_coefficients_calc():

    cdef double C, B, Z
    cdef int X

    rand_rand = random.random  # local for speed
    rand_tri = random.triangular  # local for speed

    C = 1 if rand_rand() < 0.4 else rand_tri(0, 2, 1)
    B = 1 if rand_rand() < 0.3 else rand_tri(0, 2, 1)
    Z = 0 if rand_rand() < 0.4 else rand_tri(-2, 2, 0)

    if rand_rand() < 0.5:
        C = -C

    if rand_rand() < 0.5:
        B = -B

    # Generate X (want > 0 as zero exponent kills usefullness of B and Z values as they go to 1)
    if rand_rand() < 0.4:
        X = 1
    elif rand_rand() < 0.75:
        X = 2
    else:
        X = 3

    return C, B, Z, X
