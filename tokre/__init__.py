import os
from tokre.setup import setup
from tokre.core.parsing import parse, escape
from tokre.core.tree_to_matcher import tree_to_matcher, script_to_matcher
from tokre.core.synth_feat import SynthFeat, compile

from tokre.core.matchers import EmbedData, PredData, PartialMatch, Mixer
from tokre.utils import get_vocab

from tokre.labelling.create_label import create_label
from tokre.labelling.literal_set_utils import load_literal_set, save_literal_set
from tokre.labelling.get_words import get_word_counts

# See https://stackoverflow.com/a/35904211/10222613
# Avoids import cycle issues
import sys

this = sys.modules[__name__]

this._tokenizer = None
this._workspace = None
this._openai_api_key = os.environ.get("OPENAI_API_KEY", None)


def get_tokenizer():
    if this._tokenizer is None:
        raise ValueError(
            "Tokenizer not initialized. Call `tokre.setup(tokenizer=your_tokenizer)` first."
        )
    return this._tokenizer


def get_workspace():
    if this._workspace is None:
        raise ValueError(
            "Workspace not initialized. Call `tokre.setup(tokenizer=your_tokenizer, workspace='your_workspace')` first."
        )
    return this._workspace


import transformers
import tokenizers


def encode(text):
    tokenizer = get_tokenizer()

    enc_kwargs = (
        {"add_special_tokens": False}
        if isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                tokenizers.Tokenizer,
            ),
        )
        else {}
    )

    return tokenizer.encode(text, **enc_kwargs)


enc = encode


def decode(tokens):
    tokenizer = get_tokenizer()
    return tokenizer.decode(tokens)


dec = decode


def tok_split(s: str):
    tok_ids = encode(s)
    tok_strs = [decode([tok_id]) for tok_id in tok_ids]
    return tok_strs
