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
