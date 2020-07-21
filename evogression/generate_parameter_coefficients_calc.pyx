'''
Cython implementation of EvogressionCreature._generate_parameter_coefficients()
'''
import cython
from libc.stdlib cimport rand, RAND_MAX
from random import triangular


cpdef (double, double, double, int) generate_parameter_coefficients_calc():

    cdef double C, B, Z
    cdef int X

    rand_tri = triangular  # local for speed

    # Using C random [0, 1) float via rand() / MAX_RAND is faster than Python random.random()
    C = 1 if rand_float() < 0.4 else rand_tri(0, 2, 1)
    B = 1 if rand_float() < 0.3 else rand_tri(0, 2, 1)
    Z = 0 if rand_float() < 0.4 else rand_tri(-2, 2, 0)


    if rand_float() < 0.5:
        C = -C

    if rand_float() < 0.5:
        B = -B

    # Generate X (want > 0 as zero exponent kills usefullness of B and Z values as they go to 1)
    if rand_float() < 0.4:
        X = 1
    elif rand_float() < 0.75:
        X = 2
    else:
        X = 3

    return C, B, Z, X


@cython.cdivision(True)  # avoid Python ZeroDivisionError checking (FASTER!)
cdef double rand_float():
    cdef double rand_max = RAND_MAX  # casting to double results in decimal/double result from below division (instead of 0 due to integer division)
    return rand() / rand_max
