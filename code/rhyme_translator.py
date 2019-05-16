import numpy as np
from enum import Enum
from collections import namedtuple


from translation_model_interface import ITranslationModel
from rhyme_testers import RuReversedSuffixRhymeTester, RuReversedWordRhymeTester


def get_reversed(line):
    return ''.join(reversed(line))


RHYME_DEBUG_PRINT = False # 1, 2 or 3 for more debug info

def set_rhyme_debug_print(level):
    global RHYME_DEBUG_PRINT
    RHYME_DEBUG_PRINT = level

class RhymeType(Enum):
    SUFFIX = 1
    WORD = 2


class RhymeTranslator(object):

    def __init__(self, model: ITranslationModel, rupo_engine = None):

        self._model = model
        self._out_tokenizer = model.get_output_tokenizer()
        self._out_voc = self._out_tokenizer._vocab

        self._suffix_rhyme_tester = RuReversedSuffixRhymeTester()
        self._word_rhyme_tester = RuReversedWordRhymeTester(rupo_engine = rupo_engine)


    def _model_tokens_to_line(self, output, eos_as_space):
        if type(output) == list:
            output = np.array(output)
        [line] = self._out_tokenizer.matrix_to_lines(output[np.newaxis])
        if eos_as_space and output[-1] == self._out_voc.eos_ix:
            line = ' ' + line
        return line

    def _model_tokens_to_lines(self, outputs):
        return self._out_tokenizer.matrix_to_lines(outputs)

    def _apply_softmax(self, logits, temperature):

        if temperature != 1:
            if temperature < 1:
                # Convert to 64-bit float to avoid overflows:
                # Note: There will still be overflows for T < 0.03
                logits = logits.astype(np.float64)

            logits /= temperature

        np.exp(logits, out = logits)
        logits /= logits.sum(axis = -1)[..., np.newaxis]
        return logits


    def translate_lines(self,
                        lines,
                        sample_temperature = 0,
                        max_len = 100):

        state = self._model.make_initial_state(lines)

        outputs = np.empty((len(lines), max_len + 1), dtype = np.int64)
        outputs[:, 0] = self._out_voc.bos_ix
        finished = np.zeros((len(lines),), dtype = bool)

        for t in range(max_len):

            state, logits = self._model.get_next_state_and_logits(state, outputs[:, :t + 1])

            if sample_temperature:
                # Sample from softmax with temperature:
                logits = self._apply_softmax(logits, sample_temperature)
                next_tokens = np.array([np.random.choice(len(probs), p = probs) for probs in logits])
            else:
                next_tokens = np.argmax(logits, axis = -1)

            outputs[:, t + 1] = next_tokens
            finished |= next_tokens == self._out_voc.eos_ix

            if finished.sum() == len(lines):
                break # Early exis if all lines finished

        return self._model_tokens_to_lines(outputs)



    def _translate_lines_in_rhyme(self,
                                  lines,
                                  rhyme_tester,
                                  sample_temperature,
                                  max_len,
                                  rhyme_test_counts,
                                  max_total_rhyme_tests):
        # rhyme_test_counts = (2, 3, 4) means that first token is variated 2 variants
        # second is variated 3 variants and third is variated 4 variants

        initial_state = self._model.make_initial_state(lines)
        initial_states = [[data[i: i+1] for data in initial_state] for i in range(len(lines))]

        outputs = np.empty((len(lines), max_len + 1), dtype = np.int64)
        outputs[:, 0] = self._out_voc.bos_ix
        finished = np.zeros((len(lines),), dtype = bool)

        # Rhyme states:
        # * True - rhyme found,
        # * False - there is no rhyme,
        # * None - not sure yet
        rhyme_state = None

        GenState = namedtuple('GenState', ['state', 'toks', 'prob', 'next_states'])

        gen_states = [GenState(state,
                               toks = [self._out_voc.bos_ix],
                               prob = 1,
                               next_states = [None] * rhyme_test_counts[0]) for state in initial_states]

        def fill_next_states(gen_state, t):

            last_state = t == len(rhyme_test_counts)
            if last_state:
                assert gen_state.next_states is None
                line_last_gen_states.append(gen_state)
                return

            test_count = rhyme_test_counts[t]
            assert test_count == len(gen_state.next_states)

            state, logits = self._model.get_next_state_and_logits(gen_state.state, [gen_state.toks])
            [probs] = self._apply_softmax(logits, temperature = 1) # TODO set temperature

            best_line_tokens = np.argpartition(probs, kth = -test_count, axis = -1)[-test_count:]
            best_line_token_probs = probs[best_line_tokens]

            for i in range(test_count):
                next_gen_state = GenState(state,
                                          toks = gen_state.toks + [best_line_tokens[i]],
                                          prob = gen_state.prob * best_line_token_probs[i],
                                          next_states = [None] * rhyme_test_counts[t + 1]
                                                        if t + 1 < len(rhyme_test_counts)
                                                        else None)
                gen_state.next_states[i] = next_gen_state
                fill_next_states(next_gen_state, t + 1)

        if RHYME_DEBUG_PRINT >= 2:
            print('*** DEBUG: Generating {} x {} states... ***'.format(len(lines), rhyme_test_counts)) # DEBUG
        last_gen_states = []
        for gen_state in gen_states:
            line_last_gen_states = []
            fill_next_states(gen_state, t = 0)
            last_gen_states.append(line_last_gen_states)

        # by this moment we have tree-structure (stored in last_gen_states) for each line

        assert [len(line_last_gen_states) == np.prod(rhyme_test_counts) for line_last_gen_states in last_gen_states]

        if RHYME_DEBUG_PRINT >= 3:
            for i, line_last_gen_states in enumerate(last_gen_states):
                print('*** DEBUG: Line {} suffixes: ***'.format(i + 1)) # DEBUG
                for line_last_gen_state in line_last_gen_states:
                    suffix = self._model_tokens_to_line(line_last_gen_state.toks,
                                                        eos_as_space = True)
                    print('*** DEBUG:  line {}: "{}" ***'.format(i + 1, suffix)) # DEBUG

        if RHYME_DEBUG_PRINT >= 2:
            print('*** DEBUG: Generating state pairs... ***') # DEBUG
        assert len(lines) == 2
        last_gen_state_pairs = []
        for line_1_last_gen_state in last_gen_states[0]:
            for line_2_last_gen_state in last_gen_states[1]:

                pair = (line_1_last_gen_state.prob * line_2_last_gen_state.prob,
                        line_1_last_gen_state,
                        line_2_last_gen_state)
                last_gen_state_pairs.append(pair)
        # pair is actually a triple: prob, state1, state2

        last_gen_state_pairs.sort(key = lambda t: t[0], reverse = True)
        if max_total_rhyme_tests:
            last_gen_state_pairs = last_gen_state_pairs[:max_total_rhyme_tests]

        if RHYME_DEBUG_PRINT >= 2:
            print('*** DEBUG: Testing state pairs... ***') # DEBUG
        for _, line_1_last_gen_state, line_2_last_gen_state in last_gen_state_pairs:

            state = [np.concatenate((a, b), axis = 0)
                     for a, b in zip(line_1_last_gen_state.state, line_2_last_gen_state.state)]

            outputs[0, :len(rhyme_test_counts) + 1] = line_1_last_gen_state.toks
            outputs[1, :len(rhyme_test_counts) + 1] = line_2_last_gen_state.toks

            rhyme_state = None

            def update_rhyme_state(t):
                # it will say whether we have a rhyme currently
                nonlocal rhyme_state

                line_1 = self._model_tokens_to_line(outputs[0, :t + 1],
                                                    eos_as_space = True)
                line_2 = self._model_tokens_to_line(outputs[1, :t + 1],
                                                    eos_as_space = True)
                info_1 = rhyme_tester.extract_rhyme_info(get_reversed(line_1))
                info_2 = rhyme_tester.extract_rhyme_info(get_reversed(line_2))
                rhyme_state = rhyme_tester.is_rhyme(info_1, info_2)

            update_rhyme_state(len(rhyme_test_counts))
            if rhyme_state == False:
                continue

            finished.fill(False)

            # And now the same generation function like in our model but with checking rhymes
            for t in range(len(rhyme_test_counts), max_len):

                state, logits = self._model.get_next_state_and_logits(state, outputs[:, :t + 1])

                if sample_temperature:
                    # Sample from softmax with temperature:
                    logits = self._apply_softmax(logits, sample_temperature)
                    next_tokens = np.array([np.random.choice(len(probs), p = probs) for probs in logits])
                else:
                    next_tokens = np.argmax(logits, axis = -1)

                outputs[:, t + 1] = next_tokens
                finished |= next_tokens == self._out_voc.eos_ix

                if rhyme_state is None:
                    update_rhyme_state(t + 1)
                    if rhyme_state == False:
                        break

                if finished.sum() == len(lines):
                    break # Early exis if all lines finished

            if rhyme_state != True:
                continue

            if RHYME_DEBUG_PRINT:
                print('*** DEBUG: Rhyme found! ***') # DEBUG
            return self._model_tokens_to_lines(outputs)

        if RHYME_DEBUG_PRINT:
            print('*** DEBUG: Failed to find rhyme. ***') # DEBUG
        return self.translate_lines(lines,
                                    sample_temperature,
                                    max_len)


    def translate_lines_with_rhyme(self,
                                   lines,
                                   rhyme_type = RhymeType.WORD,
                                   sample_temperature = 0,
                                   max_len = 100,
                                   rhyme_test_counts = (10, 10),
                                   max_total_rhyme_tests = 1000):

        if rhyme_type == RhymeType.SUFFIX:
            rhyme_tester = self._suffix_rhyme_tester
        elif rhyme_type == RhymeType.WORD:
            rhyme_tester = self._word_rhyme_tester
        else:
            assert False

        translated = []
        for pair_idx in range(len(lines) // 2):

            pair_lines = lines[pair_idx * 2 : (pair_idx + 1) * 2]

            translated += self._translate_lines_in_rhyme(pair_lines,
                                                         rhyme_tester,
                                                         sample_temperature,
                                                         max_len,
                                                         rhyme_test_counts,
                                                         max_total_rhyme_tests)

        if len(lines) % 2 == 1:
            translated.append(self.translate_lines(lines[-1:]
                                                   sample_temperature = sample_temperature,
                                                   max_len = max_len)[0])

        return translated
