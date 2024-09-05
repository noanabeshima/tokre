from typing import Union
import tokre

def save_literal_set(label, desc, literals: list[Union[tuple[str], str]], vary_leading_space=False, vary_capitalization=False, strip_and_lowercase=False, metadata=None):
    if metadata is None:
        metadata = {}
    assert not ((vary_leading_space or vary_capitalization) and strip_and_lowercase)

    if strip_and_lowercase is True:
        literals = literals if isinstance(literals[0], str) else [''.join(toks) for toks in literals]
        literals = [literal.strip().lowercase() for literal in literals]

    if isinstance(literals[0], (tuple, list)):
        literal_strs = [''.join(literal) for literal in literals]
        literal_toks = literals
    elif isinstance(literals[0], str):
        literal_strs = literals
        literal_toks = [tokre.tok_split(literal_str) for literal_str in literal_strs]

    normalized_literals = {literal_str.strip().lower() for literal_str in literal_strs}

    if vary_leading_space is True or vary_capitalization is True:
        if vary_capitalization:
            normalized_literals = set([literal.capitalize() for literal in normalized_literals] + [literal for literal in normalized_literals])
        
        literal_toks = {tuple(tokre.tok_split(literal)) for literal in normalized_literals}
        
        if vary_leading_space:
            literal_toks = literal_toks.union({tuple(tokre.tok_split(' '+literal)) for literal in normalized_literals})
        
        literal_toks = [literal for literal in literal_toks if '[UNK]' not in literal]
        literal_strs = [''.join(literal) for literal in literal_toks]


    result = {
            "save_literal_kwargs": {
                "label": label,
                "desc": desc,
                "vary_leading_set": vary_leading_space,
                "vary_capitalization": vary_capitalization,
                "strip_and_lowercase": strip_and_lowercase,
            },
            "literal_toks": literal_toks,
            "literal_strs": literal_strs,
            "pattern": r"(" + "|".join(literal_strs) + r")",
            "metadata": metadata
        }

    ws = tokre.get_workspace()
    tokre.utils.save_dict(result, ws / (label + '.json'))