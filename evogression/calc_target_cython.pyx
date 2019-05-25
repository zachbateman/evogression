'''
Cython implementation of EvogressionCreature._calc_single_layer_target()
'''


cpdef double calc_target_cython(dict parameters, dict modifiers, list layer_list):

    cpdef double T = 0
    T = -99999  # bogus value for first layer
    for layer in layer_list:
        T = calc_single_layer_target_cython(parameters, modifiers, layer, T)
    return T



cdef double calc_single_layer_target_cython(dict parameters, dict modifiers, long layer, double previous_T):

    cpdef double T = 0
    cpdef str param
    cpdef double value
    layer_modifiers = modifiers[f'LAYER_{layer}']

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

    cpdef dict mods

    try:
        mods = layer_modifiers[param]
        return mods['C'] * (mods['B'] * value + mods['Z']) ** mods['X']
    except KeyError:  # if param is not in self.modifiers[layer_name]
        return 0
    except ZeroDivisionError:  # could occur if exponent is negative
        return 0
    except OverflowError:
        return 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)