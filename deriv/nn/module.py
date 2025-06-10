class Parameter:
    """
    A wrapper for tensors that should be considered trainable parameters.

    Attributes:
        data (array): The underlying tensor (or array) that requires gradient updates.
    """

    def __init__(self, data):
        """
        Initialize a Parameter.

        Args:
            data (array): A tensor that will be treated as a trainable parameter.
        """
        self.data = data


class Module:
    """
    Base class for all neural network modules.

    Modules can contain parameters and other submodules. When subclassed, users should
    implement the `forward` method to define the computation.

    Attributes:
        _parameters (dict): A dictionary of named `Parameter` instances.
        _modules (dict): A dictionary of named `Module` instances.

    Methods:
        parameters(prefix=""): Recursively collects all parameters in this module and submodules.
        __call__(*args, **kwargs): Invokes the `forward` method.
        forward(*args, **kwargs): Should be implemented in subclasses to define computation.
    """

    def __init__(self):
        """
        Initialize an empty module.
        """
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        """
        Set an attribute and register it as a parameter or submodule if applicable.

        Args:
            name (str): Attribute name.
            value (Any): Value to set. If it is a `Parameter` or `Module`, it will be registered.
        """
        if isinstance(value, Parameter):
            self._parameters[name] = value.data
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self, prefix=""):
        """
        Collect all parameters in the module and its submodules.

        Args:
            prefix (str): Prefix to prepend to parameter names (used for submodules).

        Returns:
            dict: A dictionary mapping parameter names to their data tensors.
        """
        params = {}
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            params[full_name] = param
        for name, module in self._modules.items():
            subprefix = f"{prefix}.{name}" if prefix else name
            params.update(module.parameters(prefix=subprefix))
        return params

    def __call__(self, *args, **kwargs):
        """
        Call the module on inputs by delegating to `forward`.

        Returns:
            The result of `self.forward(*args, **kwargs)`.
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Defines the computation performed at every call.

        This method should be overridden by all subclasses.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Implement forward() in subclass.")
