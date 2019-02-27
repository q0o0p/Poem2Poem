import os
import sys
import getopt
import numpy as np
from collections import Counter

from vocab import Vocab



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
                     [--train-epochs=<epoch count>]
                     [--test-file=<test file>]

Train file must be UTF-8 encoded text file where each
line specifies token sequence pair separated with "|".
Tokens in each sequence are separated with whitespace.

If not specified, train epoch count is 1.

If test file is specified, it must be UTF-8 encoded
text file where each line specifies token sequence.
Tokens in each sequence are separated with whitespace.

If test file is not specified, standard input is used
for inference.

Test line tokens not present in train data are ignored
during inference.
'''.strip(),
          file = sys.stderr)
    sys.exit(1)

try:
    options, args = getopt.getopt(sys.argv[1:],
                                  shortopts = '',
                                  longopts = ['train-file=',
                                              'train-epochs=',
                                              'test-file='])
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

train_epochs = options_dict.get('--train-epochs', '1')
if train_epochs == '' or not all('0' <= ch <= '9' for ch in train_epochs):
    usage_error('"--train-epochs" option must specify a decimal number if present, not "{}".'
                .format(train_epochs))

train_epochs = int(train_epochs)
if train_epochs == 0:
    usage_error('"--train-epochs" option must not be zero.')

test_file = options_dict.get('--test-file')
if test_file is not None and not os.path.isfile(test_file):
    usage_error('"--test-file" option must specify existing file if present.')


# Read files
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

if test_file is not None:
    print('Reading test lines...')

    with open(test_file, encoding = 'utf-8') as f:
        test_lines = tuple(line.rstrip('\n') for line in f)

    print(' {} lines read.'.format(len(test_lines)))


# Tokenize train file
# --------------------

print('Tokenizing train file...')

train_toks_pairs = tuple(tuple(tuple(t.split())
                               for t in text_pair)
                         for text_pair in train_text_pairs)
del train_text_pairs

tok_ctr_pair = Counter(), Counter()
for toks_pair in train_toks_pairs:
    for tok_ctr, toks in zip(tok_ctr_pair, toks_pair):
        tok_ctr.update(toks)

for title, tok_ctr in zip(['source', 'target'], tok_ctr_pair):
    print(' {} unique tokens: {}'.format(title, len(tok_ctr)))

SOURCE_PAIR_IDX, TARGET_PAIR_IDX = 0, 1

vocab_pair = tuple(Vocab(tok_ctr.keys())
                   for tok_ctr in tok_ctr_pair)

train_tok_ids_pairs = np.array([tuple(np.array([vocab.tok_to_id[tok] for tok in toks], dtype = np.int32)
                                      for vocab, toks in zip(vocab_pair, toks_pair))
                                for toks_pair in train_toks_pairs], dtype = object)
del train_toks_pairs

if test_file is not None:
    print('Tokenizing test file...')

    test_tok_ids_seq = np.array([np.array([vocab_pair[SOURCE_PAIR_IDX].tok_to_id[tok] for tok in line.split()
                                           if tok in vocab_pair[SOURCE_PAIR_IDX].tok_to_id], dtype = np.int32)
                                 for line in test_lines], dtype = object)


# Helper functions for model training
# ----------------------------------------

def tok_ids_seq_to_matrix(tok_ids_seq, vocab):

    max_tok_count = max(map(len, tok_ids_seq))
    matrix_width = max_tok_count + 2 # For BOS and EOS

    matrix = np.full([len(tok_ids_seq), matrix_width],
                      fill_value = vocab.eos_id,
                      dtype = np.int32)
    matrix[:, 0] = vocab.bos_id

    for row_idx, tok_ids in enumerate(tok_ids_seq):
        matrix[row_idx, 1 : 1 + len(tok_ids)] = tok_ids

    return matrix

def matrix_to_lines(matrix, vocab):

    assert np.all(matrix[:, 0] == vocab.bos_id)

    lines = []
    for tok_ids in matrix[:, 1:]:
        [eos_indices] = np.where(tok_ids == vocab.eos_id)
        if len(eos_indices) != 0:
            tok_ids = tok_ids[:eos_indices[0]]
        lines.append(' '.join(vocab.toks[id] for id in tok_ids))

    return lines

def iterate_train_minibatches(batch_size):

    N = len(train_tok_ids_pairs)
    indices = np.random.permutation(np.arange(N))

    for start in range(0, N, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_tok_ids_pairs = train_tok_ids_pairs[batch_indices]
        batch_matrix_pair = tuple(map(tok_ids_seq_to_matrix, zip(*batch_tok_ids_pairs), vocab_pair))
        yield batch_matrix_pair



# Create model and optimizer
# ------------------------------

print('Creating model and optimizer...')

print(' importing TensorFlow...')
import tensorflow as tf

print(' creating session...')
sess = tf.InteractiveSession()

print(' creating model...')
from model import Seq2SeqModel

model = Seq2SeqModel(inp_eos_id = vocab_pair[SOURCE_PAIR_IDX].eos_id,
                     inp_tok_count = len(vocab_pair[SOURCE_PAIR_IDX].toks),
                     out_bos_id = vocab_pair[TARGET_PAIR_IDX].bos_id,
                     out_eos_id = vocab_pair[TARGET_PAIR_IDX].eos_id,
                     out_tok_count = len(vocab_pair[TARGET_PAIR_IDX].toks),
                     emb_size = 128,
                     hid_size = 256)

sess.run(tf.variables_initializer(model.trainable_variables))

print(' creating loss...')
target = tf.placeholder(tf.int32, [None, None])
loss = model.compute_loss(target)

print(' creating optimizer...')
opt = tf.train.AdamOptimizer()
train_step = opt.minimize(loss, var_list = model.trainable_variables)

sess.run(tf.variables_initializer(opt.variables()))



# Train model
# --------------------

print('Begin model training...')

batch_size = 32
avg_loss_steps = 10

batches_per_epoch = (len(train_tok_ids_pairs) + batch_size - 1) // batch_size

loss_history = []
for epoch_idx in range(train_epochs):

    print(' epoch {} of {}...'.format(epoch_idx + 1, train_epochs))

    for step_idx, (input_matrix, target_matrix) in enumerate(iterate_train_minibatches(batch_size)):

        step_loss, _ = sess.run([loss, train_step], { model.input: input_matrix,
                                                      target: target_matrix })

        loss_history.append(step_loss)

        avg_step_loss = np.mean(loss_history[-avg_loss_steps:])

        print('  step {} of {} ({:.0f}%), loss: {:.2f}'
              .format(step_idx + 1,
                      batches_per_epoch,
                      100 * (step_idx + 1) / batches_per_epoch,
                      avg_step_loss) + ' ' * 5,
              end = '\r')

    assert len(loss_history) == batches_per_epoch * (epoch_idx + 1)
    print(' ' * 40, end = '\r')

    epoch_losses = loss_history[-batches_per_epoch:]
    print('  avg loss: {:.2f}'
          .format(np.mean(epoch_losses)) + ' ' * 20)

    print('  last loss: {:.2f}'.format(avg_step_loss))

print(' training finished.')



# Infer using model
# --------------------

print('Inference using model:')

# Heuristic to limit inference output token count:
max_out_tok_count = max(100, max(len(target_tok_ids)
                                 for _, target_tok_ids in train_tok_ids_pairs) * 10)

if test_file is not None:
    print(' processing test file...')

    for start in range(0, len(test_lines), batch_size):
        batch_tok_ids_seq = test_tok_ids_seq[start : start + batch_size]
        batch_matrix = tok_ids_seq_to_matrix(batch_tok_ids_seq, vocab_pair[SOURCE_PAIR_IDX])
        inferred_matrix = model.infer(batch_matrix,
                                      max_out_tok_count = max_out_tok_count)
        inferred_lines = matrix_to_lines(inferred_matrix, vocab_pair[TARGET_PAIR_IDX])

        for tok_ids, inferred_line in zip(batch_tok_ids_seq, inferred_lines):
            print(' {} -> {}'.format(' '.join(vocab_pair[SOURCE_PAIR_IDX].toks[id] for id in tok_ids),
                                     inferred_line))

else:
    print(' processing standard input...')

    if os.name == 'posix':
        eof_info = 'Ctrl-D on POSIX'
    elif os.name == 'nt':
        eof_info = 'Ctrl-Z followed by Return on Windows'
    else:
        eof_info = 'your system manual to read how'
    print(' (send EOF to stop, use {})'.format(eof_info))

    for line in sys.stdin:

        line_toks = [tok for tok in line.split()
                     if tok in vocab_pair[SOURCE_PAIR_IDX].tok_to_id]
        line_tok_ids = np.array([vocab_pair[SOURCE_PAIR_IDX].tok_to_id[tok] for tok in line_toks], dtype = np.int32)

        line_matrix = tok_ids_seq_to_matrix(line_tok_ids[np.newaxis], vocab_pair[SOURCE_PAIR_IDX])
        inferred_matrix = model.infer(line_matrix,
                                      max_out_tok_count = max_out_tok_count)
        [inferred_line] = matrix_to_lines(inferred_matrix, vocab_pair[TARGET_PAIR_IDX])

        print(' {} -> {}'.format(' '.join(line_toks),
                                 inferred_line))

print(' processing finised.')
