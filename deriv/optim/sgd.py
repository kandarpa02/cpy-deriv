from deriv.Array.backend import get_backend


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer with momentum.

    This optimizer updates parameters using SGD with an optional momentum term.
    It maintains a velocity term for each parameter to smooth the updates.

    Attributes:
        parameters (dict): Dictionary of parameters to optimize.
        lr (float): Learning rate.
        beta (float): Momentum coefficient.
        velocities (dict): Dictionary of velocity buffers for each parameter.
    """

    def __init__(self, parameters, lr=1e-3, beta=0.9):
        xp = get_backend()
        """
        Initialize the SGD optimizer.

        Args:
            parameters (dict): A dictionary of parameters, where each value has `.data` and `.grad` attributes.
            lr (float): Learning rate. Default is 1e-3.
            beta (float): Momentum coefficient (between 0 and 1). Default is 0.9.
        """
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.velocities = {k: xp.zeros_like(v.data) for k, v in parameters.items()}

    def step(self):
        """
        Perform a single optimization step.

        This updates each parameter using its gradient and the momentum buffer.
        """
        for name, param in self.parameters.items():
            v = self.velocities[name]
            grad = param.grad

            v[:] = self.beta * v + (1 - self.beta) * grad 
            param.data[:] -= self.lr * v 

    def zero_grad(self):
        """
        Reset all gradients of the parameters to zero.

        This should be called before backpropagating the next batch.
        """
        for param in self.parameters.values():
            if param.grad is not None:
                param.grad.fill(0)
