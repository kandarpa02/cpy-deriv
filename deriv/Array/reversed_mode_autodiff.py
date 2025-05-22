import numpy as np

def _backward(self):
        if self.grad is None or np.all(self.grad == 0):
            self.grad = np.ones_like(self.data)
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


