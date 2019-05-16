import numpy as np



def _get_reversed(line: str) -> str:
    return ''.join(reversed(line))



class Vocab:
    def __init__(self, regular_tokens, reversed_: bool, chars: bool):

        if chars:
            bos, eos = '^', '$'
        else:
            bos, eos = '<BOS>', '<EOS>'

            if reversed_:
                bos, eos = map(_get_reversed, (bos, eos))

        assert all([t not in (bos, eos) and len(t) > 0 for t in regular_tokens])

        tokens = [bos, eos] + regular_tokens

        self.tokens_separator = '' if chars else ' '
        self.tokens = tokens
        self.token_to_ix = {t:i for i, t in enumerate(tokens)}
        self.bos_ix, self.eos_ix = 0, 1
        self._reversed = reversed_

    def __len__(self):
        return len(self.tokens)

    def tokens_to_ids(self, tokens):
        return np.array([self.token_to_ix[tok]
                         for tok in tokens])

    def token_ids_to_str(self, tok_ids: np.ndarray):
        assert type(tok_ids) == np.ndarray, 'NDarray expected as "tok_ids"'

        return self.tokens_separator.join(self.tokens[i] for i in tok_ids)


    def tok_ids_seq_to_matrix(self, tok_ids_seq, max_matrix_width):
        """
        Convert variable length token ID sequences into fixed size matrix
        If 'max_matrix_width' is not 'None', matrix is truncated
        """
        max_tok_count = max(map(len, tok_ids_seq))
        matrix_width = max_tok_count + 2 # For BOS and EOS

        if max_matrix_width is not None and matrix_width > max_matrix_width:
            assert max_matrix_width >= 1 # For BOS, EOS can be truncated
            matrix_width = max_matrix_width
            max_tok_count = matrix_width - 1

        matrix = np.full((len(tok_ids_seq), matrix_width),
                         self.eos_ix,
                         dtype = np.int32)
        matrix[:, 0] = self.bos_ix

        for i, tok_ids in enumerate(tok_ids_seq):
            row = tok_ids[:max_tok_count]
            matrix[i, 1 : 1 + len(row)] = row

        return matrix

    def matrix_to_tok_ids_seq(self, matrix):
        """
        Convert matrix of token ids into variable length token ID sequences
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        """

        assert np.all(matrix[:, 0] == self.bos_ix)

        tok_ids_seq = []
        for tok_ids in matrix[:, 1:]:
            [eos_indices] = np.where(tok_ids == self.eos_ix)
            if len(eos_indices) != 0:
                tok_ids = tok_ids[:eos_indices[0]]
            tok_ids_seq.append(tok_ids)
        return tok_ids_seq
