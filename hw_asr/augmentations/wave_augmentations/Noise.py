from hw_asr.augmentations.base import AugmentationBase
from torch import distributions

from hw_asr.augmentations.random_apply import RandomApply


class GaussianNoise(AugmentationBase):
    def __init__(self, mu: float, sigma: float, prob: float, *args, **kwargs):
        self.noiser = distributions.Normal(mu, sigma)
        self.apply = RandomApply(self.func_call, prob)

    def func_call(self, data):
        return data + self.noiser.sample(data.size())

    def __call__(self, data, *args, **kwargs):
        return self.apply(data)
