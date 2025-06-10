from deriv.Array.backend import get_backend

class SGD:
    """
    Fixed SGD optimizer that works with deriv.array and Parameter objects
    """
    
    def __init__(self, parameters, lr=1e-3, beta=0.9):
        """
        Initialize SGD optimizer
        
        Args:
            parameters (dict): Dictionary of Parameter objects
            lr (float): Learning rate
            beta (float): Momentum coefficient
        """
        self.xp = get_backend()
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        
        self.velocities = {}
        for name, param in parameters.items():
            self.velocities[name] = self.xp.zeros_like(param.data.data)

    def step(self):
        """Perform a single optimization step"""
        for name, param in self.parameters.items():
            arr = param.data
            
            if arr.grad is None:
                print(f"[WARN] No gradient for {name} — skipping update")
                continue
                
            v = self.velocities[name]
            
            grad = arr.grad
            
            v[:] = self.beta * v + (1 - self.beta) * grad
            
            # Update parameter data: w = w - lr*v
            arr.data[:] -= self.lr * v

    def zero_grad(self):
        """Reset all parameter gradients to zero"""
        for param in self.parameters.values():
            arr = param.data
            if arr.grad is not None:
                arr.grad.fill(0)