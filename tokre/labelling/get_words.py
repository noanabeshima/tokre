from tokre import SynthFeat
from typing import Iterable
import torch
import tokre



def get_word_counts(dataloader: Iterable[list[list[str]]]):
    word_synth = SynthFeat( r'''
valid_last_tok = (?<![re `[.,?!"';:#$^&*()-]|\[UNK\]` search=True])
space_tok = [re ` [\S]+`][valid_last_tok]
nospace_tok = [re `[\S]+`][valid_last_tok]
capitalized_nospace_tok = [re `[A-Z].*`]

nospace_word = (?<=\n|[re `["]` search=True])[capitalized_nospace_tok][nospace_tok]*(?=[space_tok])
space_word = [space_tok][nospace_tok]*(?=[space_tok])

word = [nospace_word] | [space_word]

[word]
''')
    
    word_counts = {}
    
    for docs in dataloader:
        per_doc_matches = word_synth.get_matches(docs)
        words = [tuple(doc[match.start:match.end].tolist()) for doc, doc_matches in zip(docs, per_doc_matches) for match in doc_matches]


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

def save_literal_set(literals: list[tuple[str]]):
    ws = tokre.get_workspace()