import pytest
import torch
from project_name.models.cnn import DemoCNN 


def test_model():
    img_sz = 32
    sample_input = torch.randn(1, 3, img_sz, img_sz)
    model = DemoCNN(img_size=img_sz, in_channels=3)
    _ = model(sample_input)



