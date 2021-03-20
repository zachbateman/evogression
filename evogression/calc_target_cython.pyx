'''
Cython implementation of EvogressionCreature._calc_single_layer_target()
'''

cpdef double calc_target_cython(dict parameters, dict modifiers):

    cpdef double T = -99999  # bogus value for first layer

    cdef str layer_name
    cdef dict layer_modifiers

    cdef double inner_T = 0
    cdef str param
    cdef double param_value
    cdef (double, double, double, int) coef
    cdef double C, B, Z
    cdef int X


    for layer_name, layer_modifiers in modifiers.items():  # DEPENDING ON ORDERED DICTIONARY HERE!!!

        inner_T = 0
        try:
            for param, param_value in parameters.items():
                if param in layer_modifiers:
                    coef = layer_modifiers[param]  # turn namedtuple into cython ctuple "coef"
                    C, B, Z, X = coef
                    inner_T += C * (B * param_value + Z) ** X

            if T != -99999:
                coef = layer_modifiers['T']  # turn namedtuple into cython ctuple "coef"
                C, B, Z, X = coef
                inner_T += C * (B * T + Z) ** X

            T = inner_T + layer_modifiers['N']

        except OverflowError:
            return 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

    return T
