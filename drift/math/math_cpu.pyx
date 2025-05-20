# cython: boundscheck=False, wraparound=False
# distutils: language=c++

cdef extern from 'math.h':
    float expf(float)
    float logf(float)
    float log10f(float)
    float tanhf(float)
    float sinf(float)
    float cosf(float)
    float tanf(float)


cdef float exp_f(float x): return expf(x)

cdef float log_f(float x): return logf(x)

cdef float log10_f(float x): return log10f(x)

cdef float tanh_f(float x): return tanhf(x)

cdef float sin_f(float x): return sinf(x)

cdef float cos_f(float x): return cosf(x)

cdef float tan_f(float x): return tanhf(x)



cdef extern from 'math.h':
    double exp(double)
    double log(double)
    double log10(double)
    double tanh(double)
    double sin(double)
    double cos(double)
    double tan(double)


cdef double exp_d(double x): return exp(x)

cdef double log_d(double x): return log(x)

cdef double log10_d(double x): return log10(x)

cdef double tanh_d(double x): return tanh(x)

cdef double sin_d(double x): return sin(x)

cdef double cos_d(double x): return cos(x)

cdef double tan_d(double x): return tanh(x)

