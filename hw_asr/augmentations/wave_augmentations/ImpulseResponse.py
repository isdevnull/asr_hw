import torch_audiomentations

from hw_asr.augmentations.base import AugmentationBase


class ImpulseResponse(AugmentationBase):
    def __init__(self, ir_path: str, prob: float, sr: int, *args, **kwargs):
        self.impulse_response = torch_audiomentations.ApplyImpulseResponse(ir_paths=ir_path, p=prob, sample_rate=sr)

    def __call__(self, data):
        return self.impulse_response(data.reshape(1, 1, -1)).squeeze(0)
