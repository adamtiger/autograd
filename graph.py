
class Graph:
    """
        Wrapper class for building networks 
        in the form of a computation graph.

        This makes the network creation similar
        to a regular dnn library.
    """
    def __init__(self):
        pass
    
    def forward(self, *params):
        raise NotImplementedError()
