import os
from tokre.setup import setup
from tokre.core.parsing import parse, escape
from tokre.core.tree_to_module import tree_to_module, compile
from tokre.core.synth_feat import SynthFeat, collect_matches

from tokre.core.modules import EmbedData, PredData, PartialMatch, Mixer
from tokre.core.pyregex import pyregex_literal

from tokre.labelling.create_label import create_label, get_all_toks

# See https://stackoverflow.com/a/35904211/10222613
# I think this avoids import cycle issues
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


def get_workspace(assert_exists=True):
    if this._workspace is None and assert_exists is True:
        raise ValueError(
            "Workspace not initialized. Call `tokre.setup(tokenizer=your_tokenizer, workspace='your_workspace')` first."
        )
    return this._workspace

import transformers
import tokenizers

def encode(text):
    tokenizer = get_tokenizer()
    
    enc_kwargs = {'add_special_tokens': False}\
        if isinstance(tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast, tokenizers.Tokenizer))\
        else {}

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

