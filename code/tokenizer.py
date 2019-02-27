class Tokenizer:

    def __init__(self, vocab):

        self.vocab = vocab


    @staticmethod
    def line_to_toks(line):

        return line.split()


    def line_to_tok_ids(self, line):

        toks = Tokenizer.line_to_toks(line)
        return self.vocab.toks_to_ids(toks)


    def tok_ids_to_line(self, tok_ids):

        return self.vocab.tok_ids_to_str(tok_ids)


    def matrix_to_lines(self, matrix):

        return [self.vocab.tok_ids_to_str(tok_ids)
                for tok_ids in self.vocab.matrix_to_tok_ids_seq(matrix)]
