from deriv.Array.backend import get_backend


def _backward(self):
    xp = get_backend()
    if self.grad is None or xp.all(self.grad == 0):
        self.grad = xp.ones_like(self.data)
    if not self._cached_topo:  
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build_topo(parent)
                self._cached_topo.append(node)
        build_topo(self)

    for node in reversed(self._cached_topo):
        node._back()

