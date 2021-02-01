from torchtext.data import Field
import torch
from onmt.inputters.vocab import SpecStayPutVocab


class BertField(Field):
    vocab_cls = SpecStayPutVocab

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="[PAD]", unk_token="[UNK]",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        super().__init__(sequential, use_vocab, init_token,
                         eos_token, fix_length, dtype,
                         preprocessing, postprocessing, lower,
                         tokenize, tokenizer_language, include_lengths,
                         batch_first, pad_token, unk_token,
                         pad_first, truncate_first, stop_words,
                         is_target)

    def process(self, batch, device=None):
        tensor = super().process(batch, device)
        if self.include_lengths:
            token_ids, lengths = tensor
        else:
            token_ids = tensor
        pad_id = torch.tensor(self.vocab.stoi[self.pad_token],
                              dtype=torch.long, device=device)
        att_mask = torch.where((token_ids != pad_id),
                               torch.tensor(1, device=device),
                               torch.tensor(0, device=device)).float()
        if self.include_lengths:
            if self.is_target:
                return (token_ids, lengths)
            else:
                return ([token_ids, att_mask], lengths)
        elif self.is_target:
            return token_ids
        else:
            return [token_ids, att_mask]
