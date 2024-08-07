import hashlib
import json
from queue import Queue
from threading import Thread

from tqdm import tqdm


def get_vocab_size(tokenizer):
    """
    Get the size of the vocabulary from the tokenizer.

    Args:
        tokenizer: The tokenizer object which should have a `vocab_size` or `n_vocab` attribute.

    Returns:
        int: The size of the vocabulary.
    """
    if hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    elif hasattr(tokenizer, "n_vocab"):
        return tokenizer.n_vocab
    else:
        raise AttributeError(
            "The tokenizer must have either 'vocab_size' or 'n_vocab' attribute."
        )


def hash_tokenizer(tokenizer):
    """
    Hash a tokenizer by deterministically hashing its vocabulary.
    Requires the tokenizer object to have:
        - a `vocab_size` or `n_vocab` attribute.
        - a `decode` method that can convert token IDs to strings.
    """
    assert hasattr(
        tokenizer, "decode"
    ), "Tokenizer must have a `decode` method for `hash_tokenizer` to work."
    assert hasattr(tokenizer, "vocab_size") or hasattr(
        tokenizer, "n_vocab"
    ), "Tokenizer must have either a `vocab_size` or `n_vocab` attribute for `hash_tokenizer` to work."

    if hasattr(tokenizer, "vocab_size"):
        assert isinstance(tokenizer.vocab_size, int)
        vocab_size = tokenizer.vocab_size  # attribute for HF tokenizers
    elif hasattr(tokenizer, "n_vocab"):
        assert isinstance(tokenizer.n_vocab, int)
        vocab_size = tokenizer.n_vocab  # attribute for tiktoken tokenizers
    else:
        raise ValueError(
            "Tokenizer must have either n_vocab or vocab_size attribute for `hash_tokenizer` to work. If your tokenizer isn't custom, are you using a HF or tiktoken tokenizer?"
        )

    # Could be sped up for >3M token tokenizers by sampling random tokens for tokens_str using a fixed random seed.
    # python hash fn is nondeterministic between sessions (!) so we use md5 instead.
    tokens_str = tokenizer.decode(list(range(vocab_size)))
    return hashlib.md5(tokens_str.encode()).hexdigest()


def threaded_map(
    fn,
    inputs,
    pbar_desc=None,
    args=None,
    kwargs=None,
    n_threads=20,
    input_is_kwargs=False,
):
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs

    pbar_desc = f"threaded_map" if pbar_desc is None else pbar_desc

    pbar = tqdm(total=len(inputs), desc=pbar_desc)
    input_queue = Queue()
    for idx, item in enumerate(inputs):
        input_queue.put((idx, item))
    results_queue = Queue()

    def worker():
        while not input_queue.empty():
            idx, item = input_queue.get()
            if input_is_kwargs:
                result = fn(*args, **item, **kwargs)
            else:
                result = fn(item, *args, **kwargs)
            results_queue.put((idx, result))
            input_queue.task_done()
            pbar.update(1)

    for _ in range(n_threads):
        thread = Thread(target=worker)
        thread.daemon = True
        thread.start()

    input_queue.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    return [result for idx, result in sorted(results, key=lambda x: x[0])]


def format_nest(nest, **kwargs):
    """
    nest is a nest of lists, dictionaries, and strings.
    returns the same nest but with the strings formatted using provided kwargs.
    """
    if isinstance(nest, list):
        return [format_nest(item, **kwargs) for item in nest]
    elif isinstance(nest, dict):
        return {
            format_nest(k, **kwargs): format_nest(v, **kwargs) for k, v in nest.items()
        }
    elif isinstance(nest, str):
        return nest.format(**kwargs)


def assert_snake_case(s):
    assert "-" not in s, "snake_case please"
    assert " " not in s, "snake_case please"
    assert s.lower() == s, "snake_case please"

def save_dict(d, fname):
    s = '{'
    for k, v in d.items():
        s += '\n  '+f'"{k}": '+json.dumps(v)+','
    s = s[:-1]
    s += '\n}\n'
    with open(fname, 'w') as f:
        f.write(s)
