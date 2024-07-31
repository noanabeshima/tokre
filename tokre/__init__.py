import os

_tokenizer = None
_workspace = None
_openai_api_key = os.environ.get("OPENAI_API_KEY", None)

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call `tokre.setup(tokenizer=your_tokenizer)` first.")
    return _tokenizer

def encode(text):
    tokenizer = get_tokenizer()
    
    return tokenizer.encode(text)
enc = encode

def decode(tokens):
    tokenizer = get_tokenizer()
    return tokenizer.decode(tokens)
dec = decode


def get_workspace(assert_exists=True):
    global _workspace
    if _workspace is None and assert_exists is True:
        raise ValueError("Workspace not initialized. Call `tokre.setup(tokenizer=your_tokenizer, workspace='your_workspace')` first.")
    return _workspace

