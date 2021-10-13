from torch import nn

from hw_asr.base import BaseModel


class BaselineLSTM(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.lstm = nn.GRU(
            input_size=n_feats,
            hidden_size=fc_hidden,
            num_layers=3,
            batch_first=True
        )
        self.proj = nn.Linear(in_features=fc_hidden, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        outputs, _ = self.lstm(spectrogram)
        return self.proj(outputs)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
