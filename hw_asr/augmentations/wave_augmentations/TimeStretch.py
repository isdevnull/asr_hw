import torch
from librosa.effects import time_stretch

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class StretchWrapper:
    def __init__(self, rate: float, *args, **kwargs):
        self.rate = rate

    def __call__(self, data):
        return time_stretch(data, rate=self.rate)


class TimeStretch(AugmentationBase):
    def __init__(self, prob: float, *args, **kwargs):
        self.prob = prob
        self.random_caller = RandomApply(StretchWrapper(*args, **kwargs), self.prob)

    def __call__(self, data, *args, **kwargs):
        return torch.from_numpy(self.random_caller(data.numpy().squeeze()))
