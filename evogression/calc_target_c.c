
#include <Python.h>
#include <math.h>
#include <stdio.h>
// See: https://www.tutorialspoint.com/python/python_further_extensions.htm


static PyObject* calc_target_c(PyObject *self, PyObject *args) {
    // arg is form (dict parameters, dict modifiers)
    double T = 0;  // bogus value for first layer
    double inner_T = 0;

    PyObject *parameters, *modifiers;

    if (!PyArg_ParseTuple(args, "OO", &parameters, &modifiers)) {
        return NULL;
    }

    PyObject *layer_name, *layer_modifiers;
    Py_ssize_t pos = 0;

    double C, B, Z;
    double X;
    PyObject *coeffs;

    PyObject *param;
    PyObject *param_value;
    Py_ssize_t pos2;

    PyObject *T_str = Py_BuildValue("s", "T");
    PyObject *N_str = Py_BuildValue("s", "N");
    const double big_num = pow(10, 150);

    // DEPENDING ON ORDERED DICTIONARY HERE!!!
    while (PyDict_Next(modifiers, &pos, &layer_name, &layer_modifiers)) {
        inner_T = 0;
        pos2 = 0;
        // while (PyDict_Next(parameters, &pos2, &param, &param_value)) {
            // if (PyDict_Contains(layer_modifiers, param)) {
                // PyArg_ParseTuple(PyDict_GetItem(layer_modifiers, param), "dddd", &C, &B, &Z, &X);
                // switch ((int) X) {  // switching for faster evaluation where possible
                    // case 0:
                        // inner_T += 1;
                    // case 1:
                        // inner_T += C * (B * PyFloat_AsDouble(param_value) + Z);
                    // default:
                        // inner_T += pow(C * (B * PyFloat_AsDouble(param_value) + Z), X);
                // }
            // }
        // }

        // if (T != 0) {  // assuming T will ONLY be zero on first run through loop... checking against zero should be faster?
            // PyArg_ParseTuple(PyDict_GetItem(layer_modifiers, T_str), "dddd", &C, &B, &Z, &X);
            // switch ((int) X) {  // switching for faster evaluation where possible
                // case 0:
                    // inner_T += 1;
                // case 1:
                    // inner_T += C * (B * T + Z);
                // default:
                    // inner_T += pow(C * (B * T + Z), X);
            // }
        // }

        // T = inner_T + PyFloat_AsDouble(PyDict_GetItem(layer_modifiers, N_str));

        while (PyDict_Next(layer_modifiers, &pos2, &param, &coeffs)) {
            if (param == N_str) {
                inner_T += PyFloat_AsDouble(coeffs);
            } else if (param == T_str) {
                PyArg_ParseTuple(coeffs, "dddd", &C, &B, &Z, &X);
                switch ((int) X) {  // switching for faster evaluation where possible
                    case 0:
                        inner_T += 1;
                    case 1:
                        inner_T += C * (B * T + Z);
                    default:
                        inner_T += pow(C * (B * T + Z), X);
                }
            } else {
                PyArg_ParseTuple(coeffs, "dddd", &C, &B, &Z, &X);
                switch ((int) X) {  // switching for faster evaluation where possible
                    case 0:
                        inner_T += 1;
                    case 1:
                        inner_T += C * (B * PyFloat_AsDouble(PyDict_GetItem(parameters, param)) + Z);
                    default:
                        inner_T += pow(C * (B * PyFloat_AsDouble(PyDict_GetItem(parameters, param)) + Z), X);
                }
            }
        }
        T = inner_T;
    }

    if (T > big_num) {
        printf("Value of T: %f\n", T);
        printf("Value of inner_T: %f\n", inner_T);
        PyErr_SetString(PyExc_OverflowError, "Overflow... T got too big! (?)");  // really big number should make this creature die if crazy bad calculations (overflow)
        return NULL;
    }

    return Py_BuildValue("d", T);  // "d" instructs to build a Python double/float
}


static PyMethodDef methods[] = {
    {"calc_target_c", &calc_target_c, METH_VARARGS, "C version of calc_target"},
    {NULL, NULL, 0, NULL}
};


// See: http://python3porting.com/cextensions.html
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "calc_target_c",  /* m_name  */
     "C version of calc_target",  /* m_doc */
    -1,                  /* m_size */
    methods,    /* m_methods */
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