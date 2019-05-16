import numpy as np

from vocab import Vocab
from tokenizer import _str_reorder


class CharVocab(Vocab):

    def __init__(self, chars, reversed_):

        super().__init__(regular_tokens = chars,
                         reversed_ = reversed_,
                         chars = True)

    def str_to_tok_ids(self, string):

        return [self.token_to_ix[ch] for ch in string]

    def lines_to_matrix(self, lines, max_matrix_width):

        tok_ids_seq = [self.str_to_tok_ids(s) for s in lines]
        return self.tok_ids_seq_to_matrix(tok_ids_seq, max_matrix_width)

    def tok_matrix_to_char_matrix(self, tok_voc, tok_matrix, max_matrix_depth):
        # input: tok_matrix is numpy array of shape (n_lines, >=max_doc_len_in_bpe_tokens)
        # input example for tok_voc.tokens = {101: 'a', 102: 'abc', 103: 'ca', ...}:
        # [[101, 102]
        #  [103, <eos>]]
        # output_example for self.tokens = {1: 'a', 2: 'b', 3: 'c', ...}:
        # [[[1, <eos>, <eos>], [1, 2, 3]]
        #  [[3, 1, , <eos>], [<eos>, <eos>, <eos>]]]
        assert tok_matrix.ndim == 2

        # First case to accept 'tok_matrix' of previous tokens only during decoding:
        assert tok_matrix.shape[1] == 1 or np.all(tok_matrix[:, 0] == tok_voc.bos_ix)

        get_tok_str = lambda tok: tok_voc.tokens[tok] if tok not in (tok_voc.bos_ix, tok_voc.eos_ix) else ''

        max_tok_len = max(len(get_tok_str(tok)) for tok in tok_matrix.flatten())
        matrix_depth = max_tok_len + 2 # For BOS and EOS

        if max_matrix_depth is not None and max_tok_len > max_matrix_depth:
            assert max_matrix_depth >= 1 # For BOS, EOS can be truncated
            matrix_depth = max_matrix_depth
            max_tok_len = matrix_depth - 1

        matrix = np.full(tok_matrix.shape + (matrix_depth,),
                         fill_value = self.eos_ix,
                         dtype = np.int32)
        matrix[:, :, 0] = self.bos_ix

        for i, tok_seq in enumerate(tok_matrix):
            for j, tok in enumerate(tok_seq):
                tok_str = _str_reorder(get_tok_str(tok),
                                       tok_voc._reversed != self._reversed)[:max_tok_len]
                matrix[i, j, 1 : 1 + len(tok_str)] = self.str_to_tok_ids(tok_str)

        return matrix
