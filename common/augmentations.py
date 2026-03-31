import torch


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.low = gamma_range[0]
        self.high = gamma_range[1]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        gamma = torch.empty(1).uniform_(self.low, self.high).item()
        return tensor.clamp(min=1e-8).pow(gamma)


class RandomWhiteBalanceShift:
    def __init__(self, strength=0.1):
        self.strength = strength

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        scales = 1.0 + (torch.rand(3, 1, 1) * 2 - 1) * self.strength
        rgb = tensor[:3] * scales
        if tensor.shape[0] > 3:
            return torch.cat([rgb.clamp(0, 1), tensor[3:]], dim=0)
        return rgb.clamp(0, 1)
