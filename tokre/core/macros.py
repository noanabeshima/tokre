from tokre.core.modules import Embed, Mixer, VarRef, PartialMatch, randstr, toks_eq
from torch import nn


def tok_split(s):
    tok_ids = tokre.enc(s)
    return [tokre.dec([tok_id]) for tok_id in tok_ids]


def get_literal_variants(tok_literal: list[str]):
    literal_str = ("".join(tok_literal)).strip()
    variants = [tok_split(literal_str), tok_split(" " + literal_str)]
    return variants


class BEGIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = f"BEGIN:{randstr()}"

    def matches(self, toks, partial, reversed):
        if toks[partial.end] == "[BEGIN]":
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + 1,
                    defns=partial.defns,
                    data=None,
                )
            ]
        else:
            return []

class AbsPos(nn.Module):
    def __init__(self, max_pos_idx=129):
        super().__init__()
        self.name = f"pos:{randstr()}"
        self.pos_embed = Embed(max_pos_idx)

    def matches(self, toks, partial, reversed):
        return [
            PartialMatch(
                name=self.name,
                start=partial.end,
                end=partial.end,
                defns=partial.defns,
                data=self.pos_embed(partial.end),
            )
        ]



class VarVariant(nn.Module):
    def __init__(self, var_ref):
        super().__init__()
        assert isinstance(var_ref, VarRef), var_ref
        self.name = f"VarVariant:{var_ref}:{randstr()}"
        self.var_name = var_ref.var_name

    def matches(self, toks, partial, reversed):
        res = []
        if self.var_name in partial.defns:
            var_toks = partial.defns[self.var_name]
            variants = get_literal_variants(var_toks)
            for variant in variants:
                if toks_eq(toks[partial.end : partial.end + len(variant)], variant):
                    res.append(
                        PartialMatch(
                            name=self.name,
                            start=partial.end,
                            end=partial.end + len(variant),
                            defns=partial.defns,
                            data=None,
                        )
                    )
        return res



class VarVariantPrefix(nn.Module):
    def __init__(self, var_ref, max_len=128):
        super().__init__()
        assert isinstance(var_ref, VarRef) or isinstance(var_ref, str), var_ref
        self.name = f"VarVariantPrefix:{str(var_ref)}:{randstr()}"
        
        self.var_name = str(var_ref) if isinstance(var_ref, str) else var_ref.var_name
        self.var_len_and_prefix_idx = Embed((max_len + 1, max_len + 1))
        self.max_len = max_len

    def matches(self, toks, partial, reversed):
        res = []
        if self.var_name in partial.defns:
            var_toks = partial.defns[self.var_name]
            variants = get_literal_variants(var_toks)
            assert all([len(variant) <= self.max_len for variant in variants]), variants
            for variant in variants:
                for prefix_len in range(1, len(variant) + 1):
                    if toks_eq(
                        toks[partial.end : partial.end + prefix_len],
                        variant[:prefix_len],
                    ):
                        res.append(
                            PartialMatch(
                                name=self.name,
                                start=partial.end,
                                end=partial.end + prefix_len,
                                defns=partial.defns,
                                data=self.var_len_and_prefix_idx(
                                    len(variant), prefix_len
                                ),
                            )
                        )
        return res

    @property
    def pyregex(self):
        return r".{0," + str(self.max_len) + r"}"


import regex as re


class TokRegex(nn.Module):
    def __init__(self, pattern, search=False):
        super().__init__()
        self.name = f"TokRegex:{pattern}:{randstr()}"
        self.pattern = pattern
        self.search = search

    def matches(self, toks, partial, reversed):
        if partial.end == len(toks):
            return []

        tok = toks[partial.end]

        if (self.search and re.search(self.pattern, tok)) or re.fullmatch(
            self.pattern, tok
        ):
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + 1,
                    defns=partial.defns,
                    data=None,
                )
            ]
        else:
            return []

class FlexRegex(nn.Module):
    def __init__(self, pattern, search=False):
        super().__init__()
        self.name = f"FlexRegex:{pattern}:{randstr()}"
        self.pattern = pattern
        self.search = search
        self.spacing = Embed(2)
        self.capitalization = Embed(6)
        self.mixer = Mixer(2, linear=True, bilinear=True)

        

    def matches(self, toks, partial, reversed):
        if partial.end == len(toks):
            return []


        tok = toks[partial.end]

        stripped_tok = tok.strip()
        normalized_tok = stripped_tok.lower()
        

        if (self.search and re.search(self.pattern, normalized_tok)) or re.fullmatch(
            self.pattern, normalized_tok
        ):
            capitalization = 0 if len(stripped_tok) == 0 else\
                  1 if stripped_tok.islower() else\
                  2 if stripped_tok.isupper() else\
                  3 if (stripped_tok[0].isupper() and len(stripped_tok) > 1 and stripped_tok[1:].islower()) else\
                  4 if stripped_tok[0].isupper() else\
                  5
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + 1,
                    defns=partial.defns,
                    data=[self.spacing(int(tok[0] == ' ')), self.capitalization(capitalization)],
                )
            ]
        else:
            return []


class TokRegexSet(nn.Module):
    def __init__(self, pattern, search=False):
        super().__init__()
        self.name = f"TokRegexSet:{pattern}:{randstr()}"
        self.pattern = pattern
        self.search = search

        if search is True:
            self.toks = {
                tok for tok in tokre.get_all_toks() if re.search(self.pattern, tok)
            }
        else:
            self.toks = {
                tok for tok in tokre.get_all_toks() if re.fullmatch(self.pattern, tok)
            }

    def matches(self, toks, partial, reversed):
        if partial.end == len(toks) and not reversed:
            return []

        if partial.end == len(toks) and reversed:
            return []

        tok = toks[partial.end]

        if tok in self.toks:
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + 1,
                    defns=partial.defns,
                    data=None,
                )
            ]
        else:
            return []


class Prefix(nn.Module):
    def __init__(self, child_module, max_len=10):
        super().__init__()
        self.name = f"Prefix:{randstr()}"
        self.child_module = child_module
        self.max_len = max_len
        self.match_len_and_prefix_len = Embed((max_len, max_len))
        self.mixer = Mixer(2, linear=True, bilinear=True)

    def matches(self, toks, partial, reversed):
        matches = self.child_module.matches(toks, partial, reversed)
        res = []
        for match in matches:
            for prefix_end in range(match.start+1, match.end+1):
                res.append(
                    PartialMatch(
                        name=self.name,
                        start=match.start,
                        end=prefix_end,
                        defns=match.defns,
                        data=[match, self.match_len_and_prefix_len(match.end-match.start, prefix_end - match.start)]
                    )
                )
        return res


import tokre
import json

import torch
from tokre.core.modules import PartialMatch


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None


class Trie:
    def __init__(self, string_lists, values=None):
        self.root = TrieNode()
        if values is None:
            values = string_lists
        for string_list, value in zip(string_lists, values):
            self.insert(string_list, value)

    def insert(self, string_list, value):
        node = self.root
        for string in string_list:
            if string not in node.children:
                node.children[string] = TrieNode()
            node = node.children[string]
        node.is_end = True
        node.value = value

    def prefixes(self, string_list):
        result = []
        node = self.root
        for i, string in enumerate(string_list):
            if string not in node.children:
                break
            node = node.children[string]
            if node.is_end:
                result.append(node.value)
        return result


class LiteralSet(nn.Module):
    def __init__(self, literal_name):
        super().__init__()
        self.name = f"Literalset:{literal_name}:{randstr()}"

        with open(tokre.get_workspace() / (literal_name + ".json")) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "literal_toks" in data

        self.literal_set = [tuple(it) for it in data["literal_toks"]]
        self.trie = Trie(self.literal_set)

        self.reversed_trie = Trie([toks[::-1] for toks in self.literal_set])

        self.literal_idx = Embed(len(self.literal_set))

        self.mixer = Mixer(1, linear=True)

    def matches(self, toks, partial, reversed):
        trie = self.trie if reversed is False else self.reversed_trie

        res = []
        forward_toks = toks[partial.end :]

        matching_prefixes = trie.prefixes(forward_toks)
        for prefix in matching_prefixes:
            prefix = prefix if reversed is False else prefix[::-1]
            res.append(
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + len(prefix),
                    defns=partial.defns,
                    data=[self.literal_idx(self.literal_set.index(prefix))],
                )
            )

        return res
    
from frozendict import frozendict

class Redefine(nn.Module):
    def __init__(self, var_name: str, new_var_name: str, ):
        self.name = f'Redefine:{randstr()}'
        self.var_name = var_name
        self.new_var_name = new_var_name

    def matches(self, toks, partial, reversed):
        if self.var_name in partial.defns:
            new_defns = {k: v for (k, v) in partial.defns.items() if k != self.var_name}
            new_defns[self.new_var_name] = partial.defns[self.var_name]
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end,
                    defns = frozendict(new_defns),
                    data=None
                )
            ]
        else:
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end,
                    defns = partial.defns,
                    data=None
                )
            ]

        



DEFINED_MACROS = {
    "var_variant_prefix": VarVariantPrefix,
    "re": TokRegex,
    "re_tok_set": TokRegexSet,
    "literal_set": LiteralSet,
    "prefix": Prefix,
    "var_variant": VarVariant,
    "BEGIN": BEGIN,
    "redefine": Redefine,
    "pos": AbsPos,
    'flex': FlexRegex
}
