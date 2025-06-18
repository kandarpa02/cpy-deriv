from deriv.Array.backend import get_backend
from deriv.optim._internals._csgd import sgd_step

class SGD:
    def __init__(self, parameters, lr=1e-3, beta=0.9):
        self.xp = get_backend()
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.velocities = {
            name: self.xp.zeros_like(param.data.data)
            for name, param in parameters.items()
        }

    def step(self):
        sgd_step(self.parameters, self.velocities, self.lr, self.beta, self.xp)

    def zero_grad(self):
        for param in self.parameters.values():
            if param.grad is not None:
                param.grad.fill(0)
