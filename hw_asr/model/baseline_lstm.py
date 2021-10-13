from torch import nn

from hw_asr.base import BaseModel


class BaselineLSTM(BaseModel):
    def __init__(self, n_feats, n_class, hidden_f, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=n_feats,
            hidden_size=hidden_f,
            num_layers=3,
            proj_size=n_class,
            batch_first=True
        )

    def forward(self, spectrogram):
        outputs, _ = self.lstm(spectrogram)
        return outputs

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
