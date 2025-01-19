from tokre.core.matchers import (
    Toks,
    Repeat,
    Phrase,
    OrGroup,
    VarDefn,
    VarRef,
    Wildcard,
    LearnedConst,
    Lookbehind,
    Lookahead,
)
from tokre.core.macros import DEFINED_MACROS
from tokre.core.parsing import parse
from lark import Transformer
import tokre


class InsertMatchers(Transformer):
    def repeat(self, children):
        assert len(children) == 3
        child_matcher, repeat_min, repeat_max = children
        return Repeat(child_matcher=child_matcher, min=repeat_min, max=repeat_max)

    def string(self, children):
        assert len(children) == 1, children
        assert isinstance(children[0], str)
        
        toks = [tokre.dec([tok_id]) for tok_id in list(tokre.enc(children[0]))]
        return Toks(toks=toks)

    def wildcard(self, children):
        return Wildcard()

    def phrase(self, children):
        return Phrase(matchers=children)

    def or_pattern(self, children):
        return OrGroup(matchers=children)

    def var_defn(self, children):
        assert len(children) == 2
        var_name, child_matcher = children
        return VarDefn(var_name=var_name, child_matcher=child_matcher)

    def var_ref(self, children):
        assert len(children) == 1, children
        var_name = children[0]
        return VarRef(var_name=var_name)

    def lookaround(self, children):
        child_matcher, is_backward, is_neg = children
        if is_backward is True:
            return Lookbehind(child_matcher, is_neg=is_neg)
        else:
            assert is_backward is False
            return Lookahead(child_matcher, is_neg=is_neg)

    def macro(self, children):
        assert len(children) >= 1

        macro_name, children = children[0], children[1:]

        args, kwargs = [], {}

        for child in children:
            if isinstance(child, dict):
                kwargs = kwargs | child
            else:
                args.append(child)

        if macro_name in DEFINED_MACROS:
            return DEFINED_MACROS[macro_name](*args, **kwargs)
        else:
            assert False, f"macro {macro_name} not found in macros.py"


def tree_to_matcher(tree):
    matcher = InsertMatchers().transform(tree)
    return matcher


from torch import nn


def recursively_add_name_to_child_matcher(matcher):
    assert not hasattr(
        matcher, "name_to_child_matcher"
    ), "matcher already has name_to_child_matcher attribute"
    name_to_child_matcher = {}

    # [STUB] hard to read code
    def add_named_child_matchers(matcher, name_to_child_matcher):
        for child_matcher in matcher.children():
            if hasattr(child_matcher, "name"):
                assert (
                    child_matcher.name not in name_to_child_matcher
                ), "Two tokre child matchers seem to have the same name?"
                name_to_child_matcher[child_matcher.name] = child_matcher

            if isinstance(child_matcher, nn.ModuleList):
                add_named_child_matchers(child_matcher, name_to_child_matcher)

        return name_to_child_matcher

    add_named_child_matchers(matcher, name_to_child_matcher)

    matcher.name_to_child_matcher = name_to_child_matcher
    for child_matcher in matcher.children():
        recursively_add_name_to_child_matcher(child_matcher)


def script_to_matcher(script: str):
    tree = parse(script)
    matcher = tree_to_matcher(tree)
    if len(list(matcher.parameters())) == 0:
        matcher = LearnedConst(matcher)
    recursively_add_name_to_child_matcher(matcher)
    return matcher
