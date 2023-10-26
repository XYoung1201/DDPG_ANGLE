#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <math.h>

void vecSum(double C1, double C2, double* x1, double* x2, double* y, int N) {
    for (int i = 0; i < N; i++)
        y[i] = C1 * x1[i] + C2 * x2[i];
}

void func(double* y, double* dy, double A) {
    dy[0] = -cos(y[1]);
    dy[1] = (A + sin(y[1])) / y[0];
    dy[2] = -sin(y[1]) / y[0];
}

void Step(double A, double* R, double* sigma, double* q, double step) {
    double y[] = { *R,*sigma,*q };
    double temp[3];
    double s1[3];
    double s2[3];
    double s3[3];
    double s4[3];

    func(y, s1, A);
    vecSum(1, step / 2, y, s1, temp, 3);
    func(temp, s2, A);
    vecSum(1, step / 2, y, s2, temp, 3);
    func(temp, s3, A);
    vecSum(1, step, y, s3, temp, 3);
    func(temp, s4, A);
    vecSum(1, 2, s1, s2, temp, 3);
    vecSum(1, 2, temp, s3, temp, 3);
    vecSum(1, 1, temp, s4, temp, 3);
    vecSum(1, step / 6, y, temp, y, 3);

    *R = y[0];
    *sigma = y[1];
    *q = y[2];
}

static PyObject* py_Step(PyObject* self, PyObject* args) {
    double step;
    double A, R, sigma, q;
    
    if (!PyArg_ParseTuple(args, "ddddd", &A, &R, &sigma, &q, &step)) {
        return NULL;
    }

    Step(A, &R, &sigma, &q, step);

    return Py_BuildValue("ddd",R,sigma,q);
}

static PyMethodDef methods[] = {
    {"Step", py_Step, METH_VARARGS, "Run a step using the Runge-Kutta method"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "PATTACK",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_PATTACK() {
    return PyModule_Create(&module);
}
