import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    spectrograms = []
    spectrograms_length = []
    texts = []
    texts_encoded = []
    texts_encoded_length = []
    # iterate over items
    for item in dataset_items:
        squeezed_spectrogram = item["spectrogram"].squeeze(0).T
        spectrograms.append(squeezed_spectrogram)
        spectrograms_length.append(squeezed_spectrogram.shape[0])

        texts.append(item["text"])

        squeezed_text_encoded = item["text_encoded"].squeeze(0)
        texts_encoded.append(squeezed_text_encoded)
        texts_encoded_length.append(len(squeezed_text_encoded))

    # form batch
    result_batch["spectrogram"] = pad_sequence(spectrograms, batch_first=True)
    result_batch["text_encoded"] = pad_sequence(texts_encoded, batch_first=True)
    result_batch["text_encoded_length"] = torch.Tensor(texts_encoded_length).int()
    result_batch["spectrogram_length"] = torch.Tensor(spectrograms_length).int()
    result_batch["text"] = texts

    return result_batch
