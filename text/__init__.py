""" from https://github.com/keithito/tacotron """

import re
from text import cleaners
from text.symbols import symbols
import json

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

with open('resources/symbol_to_id_en.json', 'w', encoding='utf8') as output:
    json.dump(_symbol_to_id, output, indent=4)

def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word


def text_to_sequence(text, cleaner_names=["english_cleaners"], dictionary=None):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence. Example: 'as is shown by the report of the Commissioners to inquire into the state of the municipal corporations in eighteen thirty-five.'
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
    '''
    sequence = []
    space = _symbols_to_sequence(' ') # 11
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            if dictionary is not None:
                clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
                for i in range(len(clean_text)):
                    t = clean_text[i]
                    if t.startswith("{"):
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)
                    sequence += space
            else:
                sequence += _symbols_to_sequence(clean_text)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
  
    # remove trailing space
    if dictionary is not None:
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    return sequence

def text_to_sequence_zh(text, cleaner_names=["english_cleaners"], dictionary=None):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: phoneme sequence. Example: ''
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
    '''
    sequence = []
    # space = [dictionary[' ']]
    text = text.split(' ')
    for i in range(len(text)):
        # print(text)
        # try:
        sequence += [dictionary.lookup(text[i])]
        # except Exception as e:
        #     print(e)
        #     print(text[i])
        # if i and i % 2 == 0:
            # sequence += space

    # if dictionary is not None:
    #     sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    
    return sequence


def sequence_to_text_zh(sequence, dictionary=None):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in dictionary:
            s = dictionary[symbol_id]
            # Enclose ARPAbet back in curly braces:
            result += s
    return result

def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'
