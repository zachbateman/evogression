'''
Cython implementation of EvogressionCreature._calc_single_layer_target()
'''


cpdef double calc_target_cython(dict parameters, dict modifiers):

    cdef str layer_name
    cpdef double T = -99999  # bogus value for first layer
    for layer_name in modifiers:  # DEPENDING ON ORDERED DICTIONARY HERE!!!
        T = calc_single_layer_target_cython(parameters, modifiers, layer_name, T)
    return T


cdef double calc_single_layer_target_cython(dict parameters, dict modifiers, str layer_name, double previous_T):

    cdef double T = 0
    cdef str param
    cdef double param_value
    cdef dict layer_modifiers = modifiers[layer_name]
    cdef (double, double, double, int) coef

    cdef double C, B, Z
    cdef int X

    for param in parameters:
        try:
            coef = layer_modifiers[param]  # turn namedtuple into cython ctuple "coef"
            C, B, Z, X = coef
            param_value = parameters[param]  # looking up value here instead of in the "for ..." line as would do in Python is faster in Cython!!!
            T += C * (B * param_value + Z) ** X
        except KeyError:  # if param is not in self.modifiers[layer_name]
            pass
        except OverflowError:
            T += 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

    if previous_T != -99999:
        try:
            coef = layer_modifiers['T']  # turn namedtuple into cython ctuple "coef"
            C, B, Z, X = coef
            T += C * (B * previous_T + Z) ** X
        except KeyError:  # if param is not in self.modifiers[layer_name]
            pass
        except OverflowError:
            T += 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

    return T + layer_modifiers['N']
