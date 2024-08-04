import os
from tokre.setup import setup
from tokre.core.parsing import parse
from tokre.core.tree_to_module import tree_to_module, compile
from tokre.core.synth_feat import SynthFeat, collect_matches

from tokre.core.modules import EmbedData, PredData, PartialMatch, Mixer
from tokre.core.pyregex import pyregex_literal


_tokenizer = None
_workspace = None
_openai_api_key = os.environ.get("OPENAI_API_KEY", None)


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        raise ValueError(
            "Tokenizer not initialized. Call `tokre.setup(tokenizer=your_tokenizer)` first."
        )
    return _tokenizer


def get_workspace(assert_exists=True):
    global _workspace
    if _workspace is None and assert_exists is True:
        raise ValueError(
            "Workspace not initialized. Call `tokre.setup(tokenizer=your_tokenizer, workspace='your_workspace')` first."
        )
    return _workspace


def encode(text):
    tokenizer = get_tokenizer()

    return tokenizer.encode(text)


enc = encode


def decode(tokens):
    tokenizer = get_tokenizer()
    return tokenizer.decode(tokens)


dec = decode


def tok_split(s: str):
    tok_ids = encode(s)
    tok_strs = [decode([tok_id]) for tok_id in tok_ids]
    return tok_strs