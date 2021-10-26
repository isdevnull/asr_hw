from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

from itertools import groupby

from pyctcdecode import build_ctcdecoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.labels = list(self.char2ind.keys())
        self.labels[0] = ""  # for decoder empty token should be empty
        self.decoder = build_ctcdecoder(self.labels)

    def ctc_decode(self, inds: torch.tensor) -> str:
        token_inds = inds.tolist()
        return "".join([c for c in [self.ind2char[t] for t, _ in groupby(token_inds)] if c != self.EMPTY_TOK])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[str]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        beams = self.decoder.decode_beams(probs[:probs_length].numpy(), beam_width=beam_size)
        hypos = [beams[i][0] for i in range(len(beams[:10]))]
        return hypos
