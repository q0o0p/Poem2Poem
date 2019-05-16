import rupo.api
from enum import Enum
from typing import NamedTuple, Optional


def get_reversed(line):
    return ''.join(reversed(line))


class LetterType(Enum):
    NOT_LETTER = 0
    VOWEL = 1
    CONSONANT = 2
    SIGN = 3

'''
Auxiliary class providing convenient functions for retrieving Russian letter type.
'''
class RuAlphabetInfo(object):

    _ru_letters = set(''.join([chr(n) for n in range(ord('а'), ord('я') + 1)]) + 'ё')
    assert len(_ru_letters) == 33
    _ru_vowels = set('аеёийоуыэюя')
    _ru_consonants = set('бвгджзклмнпрстфхцчшщ')
    _ru_signs = set('ъь')

    assert _ru_vowels & _ru_consonants == set()
    assert _ru_vowels | _ru_consonants | _ru_signs == _ru_letters

    @staticmethod
    def get_ru_letter_type(ch: str) -> LetterType:

        if ch in RuAlphabetInfo._ru_vowels:
            return LetterType.VOWEL
        elif ch in RuAlphabetInfo._ru_consonants:
            return LetterType.CONSONANT
        elif ch in RuAlphabetInfo._ru_signs:
            return LetterType.SIGN
        else:
            return LetterType.NOT_LETTER

    @staticmethod
    def lower_and_strip_left_non_letters(line: str) -> str:

        line = line.lower()

        # Skip everything before first letter:
        while len(line) > 0 and line[0] not in RuAlphabetInfo._ru_letters:
            line = line[1:]

        return line




#class RhymeInfo(NamedTuple):
#    text: str
#    finished: bool

# Old-style for Python 3.5:
RhymeInfo = NamedTuple('RhymeInfo',
                       [('text', str), # text that will be used for testing whether rhyme is present
                        ('finished', bool) # is text complete
                       ])

class IRhymeTester(object):

    def extract_rhyme_info(self, line: str) -> RhymeInfo:
        raise Exception('Not implemented')

    def is_rhyme(self, info1: RhymeInfo, info2: RhymeInfo) -> Optional[bool]:
        # Returns True - rhyme present, False - absent, None - don't know
        raise Exception('Not implemented')




'''
Implements suffix-based rhyming.
'''
class RuReversedSuffixRhymeTester(IRhymeTester):

    def extract_rhyme_info(self, line: str) -> RhymeInfo:
        # input: line is REVERSED string
        # We iterate all vowels from the beginning of line until first consonant
        # Or all consonants until first vowel
        # SIGNs are ignored but inserted to output

        line = RuAlphabetInfo.lower_and_strip_left_non_letters(line)

        finished = False
        prefix_len = 0
        prev_ch_type = None

        for ch in line:

            ch_type = RuAlphabetInfo.get_ru_letter_type(ch)

            if ch_type == LetterType.NOT_LETTER:
                finished = True
                break

            prefix_len += 1

            if ch_type != LetterType.SIGN:
                if prev_ch_type is None:
                    prev_ch_type = ch_type
                elif prev_ch_type != ch_type:
                    finished = True
                    break

        return RhymeInfo(text = line[:prefix_len],
                         finished = finished)


    def is_rhyme(self, info1: RhymeInfo, info2: RhymeInfo) -> Optional[bool]:

        if not info1.finished or not info2.finished:
            l = min(len(info1.text), len(info2.text))
            if info1.text[:l] != info2.text[:l]:
                return False
            return None

        return info1.text == info2.text


def test_RuReversedSuffixRhymeTester():

    tester = RuReversedSuffixRhymeTester()
    for line, suffix, finished in [('пижама', 'ма', True),
                                   ('обученный', 'ный', True),
                                   ('Ихтиандр', 'андр', True),
                                   ('КНДР', 'кндр', False),
                                   ('махать', 'ать', True)]:
        for l in (line, line + '!', ' ' + line + ', '):
            info = tester.extract_rhyme_info(get_reversed(l))
            assert info.text == get_reversed(suffix)
            assert info.finished == (finished or l[0] == ' ')
            assert tester.is_rhyme(info, info) == (True if info.finished else None)

#test_RuReversedSuffixRhymeTester()




'''
Implements word-based rhyming.

It uses 'rupo' library.
'''
class RuReversedWordRhymeTester(IRhymeTester):

    _WORD_INTERNAL_CHARS = set('-.')


    def __init__(self, rupo_engine = None):

        if rupo_engine is None:
            self._rupo_engine = rupo.api.Engine(language = 'ru')
            self._rupo_engine.load(stress_model_path = RUPO_STRESS_MODEL_PATH,
                                   zalyzniak_dict = RUPO_ZALYZNIAK_DICT_PATH)
        else:
            # Assume loaded engine:
            assert rupo_engine.language == 'ru'
            self._rupo_engine = rupo_engine

        self._cache = {}


    def extract_rhyme_info(self, line: str) -> RhymeInfo:
        # line is REVERSED string
        # Here we skip everything until first letter
        # return: it TRIES to return complete word (returns part in case if line contains only part)


        line = RuAlphabetInfo.lower_and_strip_left_non_letters(line)

        is_internal_char = lambda ch: ch in RuReversedWordRhymeTester._WORD_INTERNAL_CHARS

        finished = False
        word_len = 0

        for ch in line:

            is_word_char = is_internal_char(ch) or \
                           RuAlphabetInfo.get_ru_letter_type(ch) != LetterType.NOT_LETTER

            if not is_word_char:
                finished = True
                break

            word_len += 1

        while word_len > 0 and is_internal_char(line[word_len - 1]):
            word_len -= 1

        return RhymeInfo(text = get_reversed(line[:word_len]),
                         finished = finished)


    def is_rhyme(self, info1: RhymeInfo, info2: RhymeInfo) -> Optional[bool]:

        if not info1.finished or not info2.finished:
            return None

        if info1.text == info2.text:
            return False

        cache_key = (info1.text, info2.text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        res = self._rupo_engine.is_rhyme(info1.text, info2.text)
        self._cache[cache_key] = res
        return res


def test_RuReversedWordRhymeTester(rupo_engine):

    tester = RuReversedWordRhymeTester(rupo_engine)
    augment_line = lambda l: (l, l + '!', ' ' + l + ', ', 'Мы и ' + l)

    for line, word, finished in [('серая корова', 'корова', True),
                                 ('молока, много', 'много', True),
                                 ('Ихтиандр', 'ихтиандр', False),
                                 ('КНДР', 'кндр', False),
                                 ('аб-вг', 'аб-вг', False),
                                 ('.-аб.вг.-', 'аб.вг', False)]:
        for l in augment_line(line):
            info = tester.extract_rhyme_info(get_reversed(l))
            assert info.text == word
            assert info.finished == (finished or not l.startswith(line))
            assert tester.is_rhyme(info, info) == (False if info.finished else None)

    for line1, line2, is_rhyme in [('серая корова', 'не очень здорова', True),
                                   ('молока много', 'не мало', False),
                                   ('и играть', 'и скакать', False),
                                   ('и играть', 'не играть', False)]:
        for l1 in augment_line(line1):
            for l2 in augment_line(line2):
                info1 = tester.extract_rhyme_info(get_reversed(l1))
                info2 = tester.extract_rhyme_info(get_reversed(l2))
                assert tester.is_rhyme(info1, info2) == is_rhyme

#test_RuReversedWordRhymeTester(global_rupo_engine)
