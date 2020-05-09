'''
Cython implementation of EvogressionCreature._calc_single_layer_target()
'''


cpdef double calc_target_cython(dict parameters, dict modifiers, list layer_str_list):

    cpdef double T = -99999  # bogus value for first layer
    for layer_name in layer_str_list:
        T = calc_single_layer_target_cython(parameters, modifiers, layer_name, T)
    return T



cdef double calc_single_layer_target_cython(dict parameters, dict modifiers, str layer_name, double previous_T):

    cdef double T = 0
    cdef str param
    cdef double value
    cdef dict layer_modifiers = modifiers[layer_name]

    for param, value in parameters.items():
        T += param_value_component(layer_modifiers, param, value)
    if previous_T != -99999:
        T += param_value_component(layer_modifiers, 'T', previous_T)

    try:
        T += layer_modifiers['N']
    except KeyError:
        pass

    return T



cdef double param_value_component(dict layer_modifiers, str param, double value):

    cdef dict mods
    cdef double C, B, Z, X

    try:
        mods = layer_modifiers[param]
        C = mods['C']
        B = mods['B']
        Z = mods['Z']
        X = mods['X']
        return C * (B * value + Z) ** X
    except KeyError:  # if param is not in self.modifiers[layer_name]
        return 0
    except ZeroDivisionError:  # could occur if exponent is negative
        return 0
    except OverflowError:
        return 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)
