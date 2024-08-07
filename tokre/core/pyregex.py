import numpy as np
import torch
from functools import lru_cache
import unicodedata
from tokre.core.parsing import special_tokre_chars
from typing import Iterable
import tokre


def is_valid_char(code_point):
    try:
        char = chr(code_point)
        # If the character is printable, has a name, and is not a special char, it is valid.
        if (
            char.isprintable()
            and unicodedata.name(char, None)
            and char not in special_tokre_chars
        ):
            return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return False


@lru_cache
def valid_unicode_characters():
    valid_unicode_characters = []
    for code_point in range(0x110000):  # Unicode code points range from 0 to 0x10FFFF
        if is_valid_char(code_point):
            valid_unicode_characters.append(chr(code_point))
    return valid_unicode_characters


VALID_CHARS = np.array(valid_unicode_characters())


def pyregex_literal(toks):
    if (
        isinstance(toks, Iterable)
        and not isinstance(toks, str)
        and isinstance(toks[0], str)
    ):
        encoded_toks = [tokre.encode(tok) for tok in toks]
        assert all([len(tok_ids) == 1 for tok_ids in encoded_toks])
        toks = [tok_ids[0] for tok_ids in encoded_toks]

    if isinstance(toks, np.ndarray) or isinstance(toks, list):
        toks = torch.tensor(toks)
    elif isinstance(toks, torch.Tensor):
        toks = toks.cpu()
    return "".join(VALID_CHARS[toks])
