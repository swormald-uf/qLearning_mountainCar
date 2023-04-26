import torch.nn as nn

# TODO: add any useful loss functions here


# This is a simple wrapper around nn.MSELoss
class DemoMSE:
    def __init__(self, *args, **kwds) -> None:
        self.loss = nn.MSELoss(*args, **kwds) 

    def __call__(self, *args, **kwds):
        return self.loss(*args, **kwds)

