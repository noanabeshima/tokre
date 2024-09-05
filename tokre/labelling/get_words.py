from tokre import SynthFeat
from typing import Iterable
import torch
import tokre
from tqdm import tqdm


def get_word_counts(dataset):
    '''
    Takes a transformers dataset with a text column, splits up each string into token strings using the tokre-setup tokenizer, and extracts word counts.
    '''
    word_synth = tokre.SynthFeat(
        r"""
valid_last_tok = (?<![re `[.,?!"';:\#$^&*()-]` search=True])
space_tok = [re ` [\S]+`][valid_last_tok]
nospace_tok = [re `[\S]+`][valid_last_tok]
capitalized_nospace_tok = [re `[A-Z].*`]

nospace_word = (?<=(\n)|[re `["]` search=True])[capitalized_nospace_tok][nospace_tok]*(?=[space_tok])
space_word = [space_tok][nospace_tok]*(?=[space_tok])

word = [nospace_word] | [space_word]

[word]
"""
    )

    word_counts = {}

    for item in tqdm(dataset):
        doc = item['text']
        try:
            toks = tokre.tok_split(doc)
        except Exception as e:
            print(f"Error tokenizing document: {e}")
            continue
        matches = word_synth.get_matches(toks)
        words = [
            tuple(toks[match.start : match.end])
            for match in matches
        ]

        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    words, counts = list(word_counts.keys()), torch.tensor(list(word_counts.values()))

    perm = counts.topk(k=len(counts)).indices

    words = [words[i] for i in perm]
    counts = counts[perm].tolist()

    word_counts = [(word, count) for word, count in zip(words, counts)]

    return word_counts