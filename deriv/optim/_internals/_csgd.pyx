# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

cpdef void sgd_step(dict parameters, dict velocities, double lr, double beta, object xp):
    cdef object param, grad, v, data
    cdef str name

    for name in parameters:
        param = parameters[name]
        grad = param.grad
        if grad is None:
            continue

        v = velocities[name]
        v[...] = beta * v + (1 - beta) * grad

        data = param.data
        data[...] = data - lr * v
