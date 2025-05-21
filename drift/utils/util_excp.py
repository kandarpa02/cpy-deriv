class AxisError(Exception):
    def __init__(self, actual_axis):
        self.actual_axis = actual_axis
        super().__init__(f"axis {self.actual_axis} is out of bounds")