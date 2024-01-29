class Resource:
    def __init__(self, value, x_loc, y_loc):
        self.value = value
        self.location = (x_loc, y_loc)
        
    def __str__(self):
        return f"Resource of value {self.value} at {self.location}"