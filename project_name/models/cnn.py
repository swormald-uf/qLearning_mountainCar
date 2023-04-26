import torch
import torch.nn as nn


# TODO: add models here (rename this file/model or add files as needed)

class DemoCNN(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels=3,
        out_channels=1,
        hidden_dims=[32, 64, 128, 256],
        **kwargs,
    ) -> None:
        super().__init__()

        # Build layers
        channels = in_channels
        layers = []
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(h_dim),
                ]
            )
            channels = h_dim
        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)
        self.last_linear = nn.Linear(
            hidden_dims[-1] * img_size * img_size, out_channels
        )

    def forward(self, x):
        return self.last_linear(self.layers(x))



