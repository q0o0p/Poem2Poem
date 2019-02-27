import numpy as np



class Seq2SeqDataset:

    def __init__(self,
                 tokenizer_pair,
                 train_toks_pairs,
                 test_lines):

        src_tokenizer, _ = tokenizer_pair

        self.train_tok_ids_pairs = np.array([tuple(tokenizer.vocab.toks_to_ids(toks)
                                                   for tokenizer, toks in zip(tokenizer_pair, toks_pair))
                                             for toks_pair in train_toks_pairs], dtype = object)

        if test_lines is not None:
            print('Tokenizing test file...')

            self.test_tok_ids_seq = np.array([src_tokenizer.line_to_tok_ids(line)
                                              for line in test_lines], dtype = object)
