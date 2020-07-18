'''
Cython implementation of EvogressionCreature._calc_single_layer_target()
'''


cpdef double calc_target_cython(dict parameters, dict modifiers, list layer_str_list):

    cdef str layer_name
    cpdef double T = -99999  # bogus value for first layer
    for layer_name in layer_str_list:
        T = calc_single_layer_target_cython(parameters, modifiers, layer_name, T)
    return T



cdef double calc_single_layer_target_cython(dict parameters, dict modifiers, str layer_name, double previous_T):

    cdef double T = 0
    cdef str param
    cdef double value
    cdef dict layer_modifiers = modifiers[layer_name]

    cdef double C, B, Z, X

    for param, value in parameters.items():
        try:
            C, B, Z, X = layer_modifiers[param]  # unpacked namedtuple
            T += C * (B * value + Z) ** X
        except KeyError:  # if param is not in self.modifiers[layer_name]
            pass
        except OverflowError:
            T += 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

    if previous_T != -99999:
        try:
            C, B, Z, X = layer_modifiers['T']  # unpacked namedtuple
            T += C * (B * previous_T + Z) ** X
        except KeyError:  # if param is not in self.modifiers[layer_name]
            pass
        except OverflowError:
            T += 10 ** 150  # really big number should make this creature die if crazy bad calculations (overflow)

    try:
        T += layer_modifiers['N']
    except KeyError:
        pass

    return T
