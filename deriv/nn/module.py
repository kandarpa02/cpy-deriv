class Parameter:
    def __init__(self, data):
        self.data = data

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self, prefix=""):
        params = {}
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            params[full_name] = param
        for name, module in self._modules.items():
            subprefix = f"{prefix}.{name}" if prefix else name
            params.update(module.parameters(prefix=subprefix))
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Implement forward() in subclass.")
