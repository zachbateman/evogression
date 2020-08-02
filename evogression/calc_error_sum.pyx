'''
Cython implementation for finding mean of error values for an EvogressionCreature and a group of data points.
'''


cpdef double calc_error_sum(cr_calc_target_func, list data_points, list actual_target_values, int data_length):

    cdef int i
    cdef double calc_val, actual, diff
    cdef double total = 0

    for i in range(data_length):
        calc_val = cr_calc_target_func(data_points[i])
        actual = actual_target_values[i]
        diff = calc_val - actual
        total += diff ** 2.0
        # total += (cr_calc_target_func(data_points[i]) - actual_target_values[i]) ** 2.0

    return total / data_length
