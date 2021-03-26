
#include <Python.h>
#include <math.h>
#include <stdio.h>
// See: https://www.tutorialspoint.com/python/python_further_extensions.htm


// static PyObject* calc_target_c(PyObject* self, PyObject *args) {
// static PyObject *calc_target_c(PyObject *self, PyObject *args) {
static PyObject* calc_target_c(PyObject *self, PyObject *args) {
    // arg is form (dict parameters, dict modifiers)
    double T = -99999;  // bogus value for first layer
    double inner_T = 0;


    PyObject *parameters, *modifiers;

    if (!PyArg_ParseTuple(args, "OO", &parameters, &modifiers)) {
        return NULL;
    }

    PyObject *layer_name, *layer_modifiers;
    Py_ssize_t pos = 0;

    double C, B, Z;
    // int X;
    double X;
    // double N;
    PyObject *N;
    double N_double;

    PyObject *param;
    // double param_value;
    PyObject *param_value;
    double param_value_double;
    Py_ssize_t pos2;

    PyObject *T_str = Py_BuildValue("s", "T");
    PyObject *N_str = Py_BuildValue("s", "N");

    double big_num = pow(10, 150);

    // DEPENDING ON ORDERED DICTIONARY HERE!!!
    while (PyDict_Next(modifiers, &pos, &layer_name, &layer_modifiers)) {
        inner_T = 0;
        pos2 = 0;
        while (PyDict_Next(parameters, &pos2, &param, &param_value)) {
            if (PyDict_Contains(layer_modifiers, param)) {
                // printf("In While Loop If");
                PyArg_ParseTuple(PyDict_GetItem(layer_modifiers, param), "ffff", &C, &B, &Z, &X);  // turn namedtuple into components
                param_value_double = PyFloat_AsDouble(param_value);
                inner_T += pow(C * (B * param_value_double + Z), X);
                printf("Value of inner_T: %f\n",inner_T);
            }
        }

        if (T != -99999) {
            PyArg_ParseTuple(PyDict_GetItem(layer_modifiers, T_str), "ffff", &C, &B, &Z, &X);
            inner_T += pow(C * (B * T + Z), X);
            printf("Value of inner_T: %f\n",inner_T);
        }

        N = PyDict_GetItem(layer_modifiers, N_str);
        N_double = PyFloat_AsDouble(N);
        T = inner_T + N_double;

        printf("FINAL Value of T: %f\n",T);
        if (T > big_num) {
            PyErr_SetString(PyExc_OverflowError, "Overflow... T got too big! (?)");  // really big number should make this creature die if crazy bad calculations (overflow)
            return NULL;
        }
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
    // calc_target_c,    /* m_methods */
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