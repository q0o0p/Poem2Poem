import os
import sys
import getopt
import numpy as np
from collections import Counter



# Ensure not imported
# --------------------

if __name__ != "__main__":
    print('"main" module must not be imported.', file = sys.stderr)
    sys.exit(1)


# Parse options
# --------------------

def usage_error(message):
    print('Error: {}\n'.format(message),
          file = sys.stderr)
    print('''
Usage: python main.py --train-file=<train file>

Train file must be UTF-8 encoded text file where each
line specifies token sequence pair separated with "|".
Tokens in each sequence are separated with whitespace.
'''.strip(),
          file = sys.stderr)
    sys.exit(1)

try:
    options, args = getopt.getopt(sys.argv[1:],
                                  shortopts = '',
                                  longopts = ['train-file='])
except getopt.GetoptError as err:
    usage_error(err)

if args:
    usage_error('Unexpected arguments after options: {}'
                .format(', '.join('"{}"'.format(a) for a in args)))

options_dict = dict(options)
if len(options_dict) < len(options):
    usage_error('Options must not be repeated.')

train_file = options_dict.get('--train-file')
if train_file is None:
    usage_error('"--train-file" option must be specified.')
if not os.path.isfile(train_file):
    usage_error('"--train-file" option must specify existing file.')


# Read train file
# --------------------

print('Reading train text pairs...')

train_text_pairs = []

with open(train_file, encoding = 'utf-8') as f:
    for line_idx, line in enumerate(f):

        text_pair = tuple(t.strip() for t in line.split('|'))

        if len(text_pair) == 1:
            print('Train file line {} does not contain text pair separator "|": "{}"'
                  .format(line_idx + 1, line.rstrip('\n')), file = sys.stderr)
            sys.exit(1)

        if len(text_pair) > 2:
            print('Train file line {} contains multiple text pair separators "|": "{}"'
                  .format(line_idx + 1, line.rstrip('\n')), file = sys.stderr)
            sys.exit(1)

        train_text_pairs.append(text_pair)

print(' {} text pairs read.'.format(len(train_text_pairs)))


# Tokenize train file
# --------------------

print('Tokenizing train file...')

train_toks_pairs = tuple(tuple(tuple(t.split())
                               for t in text_pair)
                         for text_pair in train_text_pairs)
del train_text_pairs

tok_ctr_pair = (Counter(), Counter())
for toks_pair in train_toks_pairs:
    for tok_ctr, toks in zip(tok_ctr_pair, toks_pair):
        tok_ctr.update(toks)

for title, tok_ctr in zip(['source', 'target'], tok_ctr_pair):
    print(' {} unique tokens: {}'.format(title, len(tok_ctr)))

special_toks = ('<BOS>', '<EOS>')
bos_tok_id, eos_tok_id = 0, 1

SOURCE_PAIR_IDX, TARGET_PAIR_IDX = 0, 1

tok_list_pair = tuple(special_toks + tuple(tok_ctr.keys())
                      for tok_ctr in tok_ctr_pair)

tok_to_id_pair = tuple({ tok: idx + len(special_toks) for idx, tok
                         in enumerate(tok_list[len(special_toks):]) }
                       for tok_list in tok_list_pair)

train_tok_ids_pairs = np.array([tuple(np.array([tok_to_id[tok] for tok in toks], dtype = np.int32)
                                      for tok_to_id, toks in zip(tok_to_id_pair, toks_pair))
                                for toks_pair in train_toks_pairs], dtype = object)
del train_toks_pairs


# Helper functions for model training
# ----------------------------------------

def tok_ids_seq_to_matrix(tok_ids_seq):

    max_tok_count = max(map(len, tok_ids_seq))
    matrix_width = max_tok_count + 2 # For BOS and EOS

    matrix = np.full([len(tok_ids_seq), matrix_width],
                      fill_value = eos_tok_id,
                      dtype = np.int32)
    matrix[:, 0] = bos_tok_id

    for row_idx, tok_ids in enumerate(tok_ids_seq):
        matrix[row_idx, 1 : 1 + len(tok_ids)] = tok_ids

    return matrix

def matrix_to_lines(matrix, tok_list):

    assert np.all(matrix[:, 0] == bos_tok_id)

    lines = []
    for tok_ids in matrix[:, 1:]:
        [eos_indices] = np.where(tok_ids == eos_tok_id)
        if len(eos_indices) != 0:
            tok_ids = tok_ids[:eos_indices[0]]
        lines.append(' '.join(tok_list[id] for id in tok_ids))

    return lines

def iterate_train_minibatches(batch_size):

    N = len(train_tok_ids_pairs)
    indices = np.random.permutation(np.arange(N))

    for start in range(0, N, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_tok_ids_pairs = train_tok_ids_pairs[batch_indices]
        batch_matrix_pair = tuple(map(tok_ids_seq_to_matrix, zip(*batch_tok_ids_pairs)))
        yield batch_matrix_pair
