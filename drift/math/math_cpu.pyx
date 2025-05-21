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


cpdef float exp_f(float x): return expf(x)

cpdef float log_f(float x): return logf(x)

cpdef float log10_f(float x): return log10f(x)

cpdef float tanh_f(float x): return tanhf(x)

cpdef float sin_f(float x): return sinf(x)

cpdef float cos_f(float x): return cosf(x)

cpdef float tan_f(float x): return tanhf(x)




cdef extern from 'math.h':
    double exp(double)
    double log(double)
    double log10(double)
    double tanh(double)
    double sin(double)
    double cos(double)
    double tan(double)


cpdef double exp_d(double x): return exp(x)

cpdef double log_d(double x): return log(x)

cpdef double log10_d(double x): return log10(x)

cpdef double tanh_d(double x): return tanh(x)

cpdef double sin_d(double x): return sin(x)

cpdef double cos_d(double x): return cos(x)

cpdef double tan_d(double x): return tanh(x)


