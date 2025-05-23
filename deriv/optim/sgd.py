import numpy as np

class SGD:
    def __init__(self, parameters, lr=1e-3, beta=0.9):
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.velocities = {k: np.zeros_like(v.data) for k, v in parameters.items()}

    def step(self):
        for name, param in self.parameters.items():
            v = self.velocities[name]
            grad = param.grad

            v[:] = self.beta * v + (1 - self.beta) * grad 
            param.data[:] -= self.lr * v 

    def zero_grad(self):
        for param in self.parameters.values():
            param.grad[:] = 0

