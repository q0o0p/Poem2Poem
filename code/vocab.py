import numpy as np



class Vocab:

    def __init__(self, regular_toks):

        special_toks = '<BOS>', '<EOS>'
        self.bos_id, self.eos_id = 0, 1

        self._toks = special_toks + tuple(regular_toks)

        self._tok_to_id = { tok: idx + len(special_toks) for idx, tok
                           in enumerate(self._toks[len(special_toks):]) }


    def __len__(self):
        return len(self._toks)


    def toks_to_ids(self, toks):

        return np.array(tuple(filter(None, (self._tok_to_id.get(tok) for tok in toks))),
                        dtype = np.int32)


    def tok_ids_to_str(self, tok_ids):

        return ' '.join(self._toks[i] for i in tok_ids)


    def tok_ids_seq_to_matrix(self, tok_ids_seq):

        max_tok_count = max(map(len, tok_ids_seq))
        matrix_width = max_tok_count + 2 # For BOS and EOS

        matrix = np.full([len(tok_ids_seq), matrix_width],
                          fill_value = self.eos_id,
                          dtype = np.int32)
        matrix[:, 0] = self.bos_id

        for row_idx, tok_ids in enumerate(tok_ids_seq):
            matrix[row_idx, 1 : 1 + len(tok_ids)] = tok_ids

        return matrix


    def matrix_to_tok_ids_seq(self, matrix):

        assert np.all(matrix[:, 0] == self.bos_id)

        tok_ids_seq = []
        for tok_ids in matrix[:, 1:]:
            [eos_indices] = np.where(tok_ids == self.eos_id)
            if len(eos_indices) != 0:
                tok_ids = tok_ids[:eos_indices[0]]
            tok_ids_seq.append(tok_ids)

        return tok_ids_seq
