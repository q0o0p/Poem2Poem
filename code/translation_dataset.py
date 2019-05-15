import numpy as np

from vocab import Vocab



class TranslationDataset:

    def __init__(self,
                 tokenizer_pair,
                 train_toks_pairs,
                 test_lines):

        src_tokenizer, _ = tokenizer_pair
        self._vocab_pair = tuple(tokenizer.vocab for tokenizer in tokenizer_pair)

        self._train_tok_ids_pairs = np.array([tuple(tokenizer.vocab.toks_to_ids(toks)
                                                    for tokenizer, toks in zip(tokenizer_pair, toks_pair))
                                              for toks_pair in train_toks_pairs], dtype = object)

        if test_lines is not None:
            print('Tokenizing test file...')

            self._test_tok_ids_seq = np.array([src_tokenizer.line_to_tok_ids(line)
                                               for line in test_lines], dtype = object)


    @property
    def train_len(self):
        return len(self._train_tok_ids_pairs)


    @property
    def max_train_target_tok_id_count(self):

        return max(len(target_tok_ids)
                   for _, target_tok_ids in self._train_tok_ids_pairs)


    def get_train_batch_count(self, batch_size):

        return (self.train_len + batch_size - 1) // batch_size


    def iterate_train_minibatches(self, batch_size):

        N = self.train_len
        indices = np.random.permutation(np.arange(N))

        for start in range(0, N, batch_size):

            batch_indices = indices[start : start + batch_size]
            batch_tok_ids_pairs = self._train_tok_ids_pairs[batch_indices]
            batch_matrix_pair = tuple(map(Vocab.tok_ids_seq_to_matrix, self._vocab_pair, zip(*batch_tok_ids_pairs)))
            yield batch_matrix_pair


    def iterate_test_minibatches(self, batch_size):

        src_vocab, _ = self._vocab_pair

        for start in range(0, len(self._test_tok_ids_seq), batch_size):

            batch_tok_ids_seq = self._test_tok_ids_seq[start : start + batch_size]
            batch_matrix = src_vocab.tok_ids_seq_to_matrix(batch_tok_ids_seq)
            yield batch_matrix
