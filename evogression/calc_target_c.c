
#include <Python.h>
// See: https://www.tutorialspoint.com/python/python_further_extensions.htm


// static PyObject* calc_target_c(PyObject* self, PyObject *args) {
static PyObject *calc_target_c(PyObject *self, PyObject *args) {
    // arg is form (dict parameters, dict modifiers)
    double T = -99999;  // bogus value for first layer
    double inner_T = 0;


    PyObject *parameters, *modifiers;

    if (!PyArg_ParseTuple(args, "OO", &parameters, &modifiers)) {
        return NULL;
    }

    PyObject *layer_name, *layer_modifiers;
    Py_ssize_t pos = 0;

    PyObject *param, *param_value;
    Py_ssize_t pos2 = 0;
    // DEPENDING ON ORDERED DICTIONARY HERE!!!
    while (PyDict_Next(modifiers, &pos, &layer_name, &layer_modifiers)) {
        inner_T = 0;
        while (PyDict_Next(parameters, &pos2, &param, &param_value)) {
            if (PyDict_Contains(layer_modifiers, param)) {
                // coef = PyDict_GetItem(layer_modifiers, param)  // turn namedtuple into cython ctuple "coef"

                // if (PyArg_UnpackTuple


                // C, B, Z, X = coef;
                // inner_T += C * (B * param_value + Z) ** X;
                inner_T += 5.0;
            }
        }

        // if T != -99999:
            // coef = layer_modifiers['T']  // turn namedtuple into cython ctuple "coef"
            // C, B, Z, X = coef
            // inner_T += C * (B * T + Z) ** X

        // T = inner_T + layer_modifiers['N']

        // if (T > 10 ** 150) {
            // return PyExc_OverflowError();  // really big number should make this creature die if crazy bad calculations (overflow)
        // }
    }

    T = inner_T + 1;
    return Py_BuildValue("d", T);  // "d" instructs to build a Python double/float
}

// See: http://python3porting.com/cextensions.html
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "calc_target_c",  /* m_name  */
     "C version of calc_target",  /* m_doc */
    -1,                  /* m_size */
    calc_target_c,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};


static PyObject *moduleinit(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    return m;
}


PyMODINIT_FUNC PyInit_calc_target_c(void) {
    return moduleinit();
}