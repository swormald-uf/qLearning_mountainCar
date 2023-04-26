# Some example image tranforms
import torchvision.transforms as transforms
import numpy as np


# TODO: add any useful transforms here, rename this file/transform or add files as needed


# Simple transform to tensor to a numpy array
class TensorImageToNumpy(object):
    def __init__(self, grayscale=False) -> None:
        mode = "L" if grayscale else "RGB"
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: np.array(x.convert(mode))),
            ]
        )

    def __call__(self, tensor):
        return self.transform(tensor)
