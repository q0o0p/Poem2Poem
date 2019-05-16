import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, NamedTuple, Optional

from vocab import Vocab
from tokenizer import Tokenizer, Lang



def _get_reversed(line: str) -> str:
    return ''.join(reversed(line))



def _open_read_text(path):
    return open(path, mode = 'r', encoding = 'utf-8', newline = '\n') # input lines are only terminated by '\n'


def _update_tok_id_pairs(tok_pairs):

    # Update 'tok_pairs' where first token ID = 0:
    for tok_pair in tqdm(tok_pairs):
        for toks in tok_pair:
            assert type(toks) == np.ndarray
            toks += 2


def _load_tok_pairs(path, tok_ids):

    tok_pairs = []

    with _open_read_text(path) as f:

        for line in tqdm(f):
            assert line[-1] == '\n'
            tok_pair = line[:-1].split(';')
            assert len(tok_pair) == 2

            tok_pair = [s.split() for s in tok_pair]

            if tok_ids:
                tok_pair = [np.array([int(s, 16) for s in toks], dtype = np.int32)
                            for toks in tok_pair]

            tok_pairs.append(tuple(tok_pair))

    return tok_pairs


def _load_text_pairs(path):

    text_pairs = []

    with _open_read_text(path) as f:
        for line in tqdm(f):
            text_pairs += [tuple(p) for p in json.loads(line)]

    return text_pairs


class AdditionalDataset(NamedTuple):
    name: str
    path: str
    reversed_: bool
    store_text: bool

class TranslationDataset(object):

    def __init__(self,
                 src_lang: Lang,
                 tgt_lang: Lang,
                 src_lower: bool,
                 tgt_lower: bool,
                 src_reversed: bool,
                 tgt_reversed: bool,
                 src_bpe_path: str,
                 tgt_bpe_path: str,
                 src_vocab_path: str,
                 tgt_vocab_path: str,
                 train_dataset_tok_ids_path: str,
                 dev_dataset_tok_ids_path: str,
                 dev_dataset_toks_path: str,
                 additional_datasets: Optional[List[AdditionalDataset]] = None):

        assert all(lang in (Lang.EN, Lang.RU) for lang in [src_lang, tgt_lang])
        assert src_lang != tgt_lang
        assert os.path.exists(src_bpe_path)
        assert os.path.exists(tgt_bpe_path)
        assert os.path.exists(src_vocab_path)
        assert os.path.exists(tgt_vocab_path)
        assert os.path.exists(train_dataset_tok_ids_path)
        assert os.path.exists(dev_dataset_tok_ids_path)
        assert os.path.exists(dev_dataset_toks_path)


        self._lang_indices = { src_lang: 0, tgt_lang: 1 }
        self._tokenizer_pair = [Tokenizer(bpe_path = src_bpe_path,
                                          vocab_path = src_vocab_path,
                                          lang = src_lang,
                                          lower = src_lower,
                                          reversed_ = src_reversed),

                                Tokenizer(bpe_path = tgt_bpe_path,
                                          vocab_path = tgt_vocab_path,
                                          lang = tgt_lang,
                                          lower = tgt_lower,
                                          reversed_ = tgt_reversed)]


        print('Reading train token ID lines...', flush = True)
        train_tok_id_pairs = _load_tok_pairs(train_dataset_tok_ids_path, tok_ids = True)
        print(' updating token IDs...', flush = True)
        _update_tok_id_pairs(train_tok_id_pairs)
        self._train_tok_id_pairs = np.array(train_tok_id_pairs, dtype = object)

        print('Reading dev token ID lines...', flush = True)
        dev_tok_id_pairs = _load_tok_pairs(dev_dataset_tok_ids_path, tok_ids = True)
        print(' updating token IDs...', flush = True)
        _update_tok_id_pairs(dev_tok_id_pairs)
        self._dev_tok_id_pairs = np.array(dev_tok_id_pairs, dtype = object)

        print('Reading dev token lines...', flush = True)
        dev_tok_pairs = _load_tok_pairs(dev_dataset_toks_path, tok_ids = False)

        print(' converting dev tokens to text...', flush = True)
        dev_text_pairs = [tuple(tokenizer.tokens_to_str(toks) for tokenizer, toks
                                in zip(self._tokenizer_pair, toks_pair))
                          for toks_pair in tqdm(dev_tok_pairs)]
        self._dev_text_pairs = np.array(dev_text_pairs, dtype = object)

        if additional_datasets:
            print('Reading additional datasets...', flush = True)
            self._additional_datasets = {}

            for dataset in additional_datasets:
                print(' Reading dataset "{}"...'.format(dataset.name), flush = True)
                text_pairs = _load_text_pairs(dataset.path)

                print('  tokenizing...', flush = True)
                tok_id_pairs = [[tokenizer.str_to_tok_ids(text, str_reversed = dataset.reversed_)
                                 for text, tokenizer in zip(text_pair, self._tokenizer_pair)]
                                for text_pair in tqdm(text_pairs)]

                dataset_dict = { 'tok_id_pairs': np.array(tok_id_pairs, dtype = object) }

                if dataset.store_text:
                    if dataset.reversed_:
                        print('  dereversing dataset...', flush = True)
                        text_pairs = [tuple(map(_get_reversed, text_pair))
                                      for text_pair in tqdm(text_pairs)]

                    dataset_dict['text_pairs'] = np.array(text_pairs, dtype = object)

                self._additional_datasets[dataset.name] = dataset_dict


    def get_tokenizer(self, lang: Lang):
        return self._tokenizer_pair[self._lang_indices[lang]]


    def _to_matrix_pair(self, tok_ids_pairs, max_matrix_width):
        tok_ids_seq_pair = zip(*tok_ids_pairs)
        matrix_pair = [tokenizer.tok_ids_seq_to_matrix(tok_ids_seq, max_matrix_width) for tokenizer, tok_ids_seq
                       in zip(self._tokenizer_pair, tok_ids_seq_pair)]
        return tuple(matrix_pair)

    def _iterate_minibatches(self, tok_id_pairs, text_pairs, shuffle, batch_size, pad_last_batch, max_batch_matrix_width, epoch_count, parallel_data):

        if pad_last_batch:
            assert shuffle

        if text_pairs is not None:
            assert len(tok_id_pairs) == len(text_pairs)

        if parallel_data is not None:
            assert len(tok_id_pairs) == len(parallel_data)

            if type(parallel_data) != np.ndarray:
                assert type(parallel_data) == list or type(parallel_data) == tuple
                parallel_data = np.array(parallel_data, dtype = object)

        def make_batch_data(matrix_pair, batch_text_pairs, batch_parallel_data):
            if text_pairs is None and parallel_data is None:
                return matrix_pair

            text_seq_pair = tuple(map(list, zip(*batch_text_pairs))) \
                            if text_pairs is not None else None

            values = (matrix_pair, text_seq_pair, batch_parallel_data)

            return tuple(v for v in values if v is not None)

        N = len(tok_id_pairs)
        epoch = 0
        while True:
            if shuffle:
                indices = np.random.permutation(np.arange(N))

                if pad_last_batch and N % batch_size > 0:
                    pad_indices = np.random.choice(N, size = batch_size - N % batch_size, replace = False)
                    indices = np.hstack([indices, pad_indices])

                for start in range(0, len(indices), batch_size):
                    batch_indices = indices[start : start + batch_size]
                    matrix_pair = self._to_matrix_pair(tok_id_pairs[batch_indices], max_batch_matrix_width)

                    batch_text_pairs = text_pairs[batch_indices] \
                                       if text_pairs is not None else None

                    batch_parallel_data = parallel_data[batch_indices] \
                                          if parallel_data is not None else None

                    yield make_batch_data(matrix_pair, batch_text_pairs, batch_parallel_data)
            else:
                for start in range(0, N, batch_size):
                    matrix_pair = self._to_matrix_pair(tok_id_pairs[start : start + batch_size], max_batch_matrix_width)

                    batch_text_pairs = text_pairs[start : start + batch_size] \
                                       if text_pairs is not None else None

                    batch_parallel_data = parallel_data[start : start + batch_size] \
                                          if parallel_data is not None else None

                    yield make_batch_data(matrix_pair, batch_text_pairs, batch_parallel_data)

            epoch += 1
            if epoch == epoch_count: # If epoch_count is 0 or None never terminate
                break

    def iterate_train_minibatches(self, batch_size, max_batch_matrix_width, epoch_count, shuffle = True, pad_last_batch = False, parallel_data = None):
        return self._iterate_minibatches(tok_id_pairs = self._train_tok_id_pairs,
                                         text_pairs = None,
                                         shuffle = shuffle,
                                         batch_size = batch_size,
                                         pad_last_batch = pad_last_batch,
                                         max_batch_matrix_width = max_batch_matrix_width,
                                         epoch_count = epoch_count,
                                         parallel_data = parallel_data)

    def iterate_dev_minibatches(self, batch_size, max_batch_matrix_width, epoch_count, shuffle = False, pad_last_batch = False, parallel_data = None):
        return self._iterate_minibatches(tok_id_pairs = self._dev_tok_id_pairs,
                                         text_pairs = self._dev_text_pairs,
                                         shuffle = shuffle,
                                         batch_size = batch_size,
                                         pad_last_batch = pad_last_batch,
                                         max_batch_matrix_width = max_batch_matrix_width,
                                         epoch_count = epoch_count,
                                         parallel_data = parallel_data)

    def iterate_additional_minibatches(self, dataset_name, batch_size, max_batch_matrix_width, epoch_count, shuffle, pad_last_batch = False, parallel_data = None):
        dataset = self._additional_datasets[dataset_name]
        return self._iterate_minibatches(tok_id_pairs = dataset['tok_id_pairs'],
                                         text_pairs = dataset.get('text_pairs'),
                                         shuffle = shuffle,
                                         batch_size = batch_size,
                                         pad_last_batch = pad_last_batch,
                                         max_batch_matrix_width = max_batch_matrix_width,
                                         epoch_count = epoch_count,
                                         parallel_data = parallel_data)
