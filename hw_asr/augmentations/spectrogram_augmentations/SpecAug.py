import torchaudio.transforms
from torch import nn

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class SpecAug(AugmentationBase):
    def __init__(self, freq_mask: int, time_mask: int, prob: float, *args, **kwargs):
        self.augmentation = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask),
            torchaudio.transforms.TimeMasking(time_mask)
        )
        self.prob = prob
        self.random_caller = RandomApply(self.augmentation, self.prob)

    def __call__(self, data, *args, **kwargs):
        return self.random_caller(data)

