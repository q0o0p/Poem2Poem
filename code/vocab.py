class Vocab:

    def __init__(self, regular_toks):

        special_toks = '<BOS>', '<EOS>'
        self.bos_id, self.eos_id = 0, 1

        self.toks = special_toks + tuple(regular_toks)

        self.tok_to_id = { tok: idx + len(special_toks) for idx, tok
                           in enumerate(self.toks[len(special_toks):]) }
