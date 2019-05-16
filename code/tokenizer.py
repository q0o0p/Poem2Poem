import os
import re
import string
import numpy as np
from enum import Enum
from tqdm import tqdm
from subword_nmt.apply_bpe import BPE

from vocab import Vocab



def _count_to_str(n: int) -> str:
    if n >= 1_000_000:
        n /= 1_000_000
        suffix = 'M'
    elif n >= 1_000:
        n /= 1_000
        suffix = 'k'
    else:
        return str(n)

    return ('{:.1f}' if n < 10 else '{:.0f}').format(n) + suffix

def _get_reversed(line: str) -> str:
    return ''.join(reversed(line))

def _str_reorder(line: str, reverse: bool) -> str:
    return _get_reversed(line) if reverse else line



def _open_read_text(path):
    return open(path, mode = 'r', encoding = 'utf-8', newline = '\n') # input lines are only terminated by '\n'


def _load_bpe(bpe_path, separator):
    with _open_read_text(bpe_path) as f: # input lines are only terminated by '\n'
        return BPE(codes = f, separator = separator)

def _remove_punc_en(line, lower):
    if lower:
        line = line.lower()
    line = re.sub(r"[^\w\s'-]|_", ' ', line) # leave '-' and "'", remove '_'

    # Leave only one '-' or "'" in row:

    line = re.sub("'['-]+", "' ", line)  # ams'- he said
    line = re.sub("['-]+'", " '", line)  # -'John' said
    line = re.sub("['-]['-]+", ' ', line) # -'- or --

    line = ' ' + line + ' '
    line = re.sub(r'-\s', ' ', line)  # leve '-' only inside words
    line = re.sub(r'\s-', ' ', line)  # same, to match 'a- -a'
    line = re.sub(r"\s'\s", ' ', line)    # remove "'" alone

    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def _remove_punc_ru(line, lower):
    if lower:
        line = line.lower()
    line = re.sub(r"[^\w\s-]|_", ' ', line) # leave '-', remove '_'

    # Leave only one '-' in row:

    line = re.sub("--+", ' ', line) # --

    line = ' ' + line + ' '
    line = re.sub(r'-\s', ' ', line)  # leve '-' only inside words
    line = re.sub(r'\s-', ' ', line)  # same, to match 'a- -a'

    line = re.sub(r'\s+', ' ', line)
    return line.strip()



class Lang(Enum):
    EN = 1
    RU = 2

#def _get_lang_idx(lang: Lang):
#    if lang == Lang.EN:
#        return 0
#    elif lang == Lang.RU:
#        return 1
#    else:
#        assert False, 'Unknown language ' + str(lang)

def _str_preprocess(s: str, lang: Lang, lower: bool, reversed_: bool):
    if lang == Lang.RU:
        s = _remove_punc_ru(s, lower = lower)
    else:
        s = _remove_punc_en(s, lower = lower)
    s = _str_reorder(s, reversed_)
    return s

_lang_char_sets = { Lang.EN: set(string.ascii_lowercase + "'-"),
                    Lang.RU: set(chr(n) for n in range(ord('а'), ord('я') + 1)) | set('ё-') }

class Tokenizer(object):

    def __init__(self,
                 bpe_path: str,
                 vocab_path: str,
                 lang: Lang,
                 lower: bool,
                 reversed_: bool):

        assert os.path.exists(bpe_path)
        assert os.path.exists(vocab_path)

        print('Loading {} tokenizer...'.format('English' if lang == Lang.EN else 'Russian'))
        self._lower = lower
        self._reversed = reversed_
        self._lang = lang

        self._char_set = _lang_char_sets[lang] | set(' ')
        if not lower:
            self._char_set |= { ch.upper() for ch in _lang_char_sets[lang] }

        print(' Loading BPE...')
        self._bpe = _load_bpe(bpe_path, separator = '@')


        print(' Reading vocabulary file...', flush = True)

        tok_list = []
        with _open_read_text(vocab_path) as voc_f:
            for tok_line in tqdm(voc_f):
                assert tok_line[-1] == '\n'
                tok_list.append(tok_line[:-1])

        print(' token count: {}'.format(_count_to_str(len(tok_list))))

        self._vocab = Vocab(regular_tokens = tok_list, reversed_ = reversed_, chars = False)


    def str_to_tok_ids(self, s: str, str_reversed: bool = False):
        s = _str_preprocess(s, self._lang, self._lower, self._reversed != str_reversed)
        assert all(ch in self._char_set for ch in s), \
            'String "{}" contains invalid characters'.format(s)
        toks = self._bpe.segment_tokens(s.split())
        return self._vocab.tokens_to_ids(toks)

    def _tokens_str_to_str(self, s: str):
        s = s.replace('@ ', '')
        return _str_reorder(s, self._reversed)

    def tokens_to_str(self, tokens):
        s = ' '.join(tokens)
        s = s.replace('@ ', '')
        return _str_reorder(s, self._reversed)

    def tok_ids_to_str(self, tok_ids: np.ndarray):
        assert type(tok_ids) == np.ndarray, 'NDarray expected as "tok_ids"'

        return self._tokens_str_to_str(self._vocab.token_ids_to_str(tok_ids))


    def tok_ids_seq_to_matrix(self, tok_ids_seq, max_matrix_width):
        return self._vocab.tok_ids_seq_to_matrix(tok_ids_seq, max_matrix_width)

    def lines_to_matrix(self, lines, max_matrix_width):

        tok_ids_seq = [self.str_to_tok_ids(s) for s in lines]
        return self.tok_ids_seq_to_matrix(tok_ids_seq, max_matrix_width)

    def matrix_to_lines(self, matrix: np.ndarray):
        """
        Convert matrix of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        """
        lines = [self.tok_ids_to_str(tok_ids)
                 for tok_ids in self._vocab.matrix_to_tok_ids_seq(matrix)]
        return lines
