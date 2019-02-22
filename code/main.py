import os
import sys
import getopt



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
