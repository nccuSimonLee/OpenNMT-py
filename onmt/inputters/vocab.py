from collections import defaultdict
from torchtext.vocab import Vocab
from ..constants import DefaultTokens, BertTokens


class SpecStayPutVocab(Vocab):
    def __init__(self, counter, max_size=None,
                 min_freq=1, specials=['<unk>', '<pad>']):
        """Create a Vocab object from a collections.Counter.

        copy from torchtext==0.5.0

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary.
                Default: ['<unk'>, '<pad>']
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(filter(lambda tup: tup[1] >= min_freq,
                                              counter.items()),
                                       key=lambda tup: tup[0])[:max_size]
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        self.stoi = {word: i
                     for i, (word, _) in enumerate(words_and_frequencies)}
        cur_idx = max(self.stoi.values()) + 1
        for sp in specials:
            if sp not in self.stoi:
                self.stoi[sp] = cur_idx
                cur_idx += 1
        self.unk_index = self.stoi.get(BertTokens.UNK,
                                       self.stoi.get(DefaultTokens.UNK, 0))
        self.stoi = defaultdict(lambda: self.unk_index, self.stoi)
        self.itos = {i: word for word, i in self.stoi.items()}

        self.vectors = None
