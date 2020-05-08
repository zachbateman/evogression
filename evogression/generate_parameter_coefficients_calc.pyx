'''
Cython implementation of EvogressionCreature._generate_parameter_coefficients()
'''
import random


cpdef tuple generate_parameter_coefficients_calc(no_negative_exponents):

    cpdef double C
    cpdef double B
    cpdef double Z
    cpdef double X

    rand_rand = random.random  # local variable for speed
    rand_tri = random.triangular
    rand_choice = random.choice

    C = 1 if rand_rand() < 0.4 else rand_tri(0, 2, 1)
    B = 1 if rand_rand() < 0.3 else rand_tri(0, 2, 1)
    Z = 0 if rand_rand() < 0.4 else rand_tri(-2, 2, 0)
    if rand_rand() < 0.5:
        C = -C
    if rand_rand() < 0.5:
        B = -B
    if no_negative_exponents:
        X = 1 if rand_rand() < 0.4 else rand_choice([0] * 1 + [2] * 5 + [3] * 2)
    else:
        X = 1 if rand_rand() < 0.4 else rand_choice([-2] * 1 + [-1] * 5 + [0] * 3 + [2] * 5 + [3] * 1)
    return C, B, Z, X
