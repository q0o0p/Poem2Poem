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
