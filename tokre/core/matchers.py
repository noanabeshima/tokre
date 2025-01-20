from __future__ import annotations
import torch
import torch.nn as nn

from typing import Union, TypeAlias, Optional
from dataclasses import dataclass
from frozendict import frozendict

from .prediction_primitives import *


def batched_extend_matches(
    toks, partial_matches: list[PartialMatch], child_matcher, reversed
):
    """
    Extends a batch of partial matches by applying a child matcher to each one.

    Args:
        toks: List of tokens to match against
        partial_matches: List of PartialMatch objects to extend
        child_matcher: Matcher module to apply to each partial match
        reversed: Whether to match in reverse direction

    Returns:
        list[PartialMatch]: List of new extended partial matches, where each original
        partial match may generate zero or more extended matches based on the child matcher results.
        The data field of each extended match appends the child match data to the original partial's data.
    """
    new_partials = []

    for partial in partial_matches:
        assert isinstance(
            partial.data, list
        ), f"Provided partial match group_data must be represented as a list: {partial.data=}"
        matcher_results = child_matcher.get_partial_match_extensions(
            toks, partial, reversed=reversed
        )

        for match_extension in matcher_results:
            extended_match = PartialMatch(
                name=partial.name,
                start=partial.start,
                end=match_extension.end,
                defns=match_extension.defns,
                data=partial.data + [match_extension],
            )
            new_partials.append(extended_match)
    return new_partials


def toks_eq(toks_a: list[str], toks_b: list[str]):
    return (len(toks_a) == len(toks_b)) and all(
        [a == b for (a, b) in zip(toks_a, toks_b)]
    )


Inf = float("inf")


class Matcher(nn.Module):
    """Base class for tokre pattern matching modules.
    Implements default get_partial_match_extensions method that returns empty list.
    All matcher modules should inherit from this class."""

    def __init__(self):
        super().__init__()

    def get_partial_match_extensions(
        self, toks: list[str], partial: PartialMatch, reversed: bool
    ):
        """Abstract method that must be implemented by subclasses.

        Args:
            toks: list[str], list of tokens to match against
            partial: PartialMatch, current partial match state
            reversed: bool, Whether to match tokens in reverse order

        Returns:
            List of PartialMatch objects
        """
        raise NotImplementedError(
            "Subclasses must implement get_partial_match_extensions"
        )


class Toks(Matcher):
    def __init__(self, toks: list[str], name: Optional[str] = None):
        super().__init__()
        self.name = "Toks:" + randstr() if name is None else name
        self.toks = toks

    def get_partial_match_extensions(self, toks, partial, reversed: bool):
        match_toks = self.toks if reversed is False else self.toks[::-1]
        if toks_eq(toks[partial.end : partial.end + len(match_toks)], match_toks):
            match = PartialMatch(
                name=self.name,
                start=partial.end,
                end=partial.end + len(match_toks),
                defns=partial.defns,
                data=None,
            )
            return [match]
        else:
            return []

    def __repr__(self):
        return f"Toks({self.toks})"


class Repeat(Matcher):
    def __init__(self, child_matcher, min: int, max: Union[int, Inf], name=None):
        super().__init__()
        self.name = "Repeat:" + randstr() if name is None else name
        self.child_matcher = child_matcher
        self.min = min
        self.max = max

        self.n_repeats = Embed(128, "n_repeats")

        if max == Inf:
            self.d_mixer = 1
        else:
            assert isinstance(max, int)
            self.n_repeats = Embed(max + 1)
            self.d_mixer = max + 1
            if self.d_mixer > 1:
                self.mixer = Mixer(self.d_mixer, linear=True)

    def get_partial_match_extensions(self, toks, partial, reversed):
        starting_partial = PartialMatch(
            name=self.name,
            start=partial.end,
            end=partial.end,
            defns=partial.defns,
            data=[],
        )

        res = []
        new_partials = [starting_partial]
        if self.min == 0:
            res.extend(new_partials)

        repeat_idx = 1
        while repeat_idx <= self.max and new_partials:
            new_partials = batched_extend_matches(
                toks, new_partials, self.child_matcher, reversed=reversed
            )
            if repeat_idx >= self.min:
                res.extend(new_partials)
            repeat_idx += 1

        # Special data if self.max == Inf
        if self.max == Inf:
            for partial in res:
                partial.data = self.n_repeats(len(partial.data))

        return res

    def extra_repr(self):
        return f"""(min): {self.min}
(max): {self.max}"""


class Phrase(Matcher):
    def __init__(self, matchers, name=None):
        super().__init__()
        self.name = "Phrase:" + randstr() if name is None else name
        self.matchers = nn.ModuleList(matchers)

        self.mixer = Mixer(len(self.matchers), linear=True, bilinear=True)

    def get_partial_match_extensions(self, toks, partial, reversed: bool):
        starting_partial = PartialMatch(
            name=self.name,
            start=partial.end,
            end=partial.end,
            defns=partial.defns,
            data=[],
        )

        partials = [starting_partial]

        for matcher in self.matchers:
            partials = batched_extend_matches(
                toks, partials, matcher, reversed=reversed
            )
        return partials


class Wildcard(Matcher):
    def __init__(self, name=None):
        super().__init__()
        self.name = "Wildcard:" + randstr() if name is None else name

    def get_partial_match_extensions(self, toks, partial, reversed: bool):
        if partial.end < len(toks):
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


class OrGroup(Matcher):
    def __init__(self, matchers, name=None):
        super().__init__()
        self.name = "OrGroup:" + randstr() if name is None else name
        self.branches = nn.ModuleList(matchers)

        self.which_branch = Embed(len(self.branches))

        self.mixer = Mixer(2, linear=True)

    def get_partial_match_extensions(self, toks, partial, reversed=False):
        res = []
        for branch_idx, branch in enumerate(self.branches):
            for match in branch.get_partial_match_extensions(
                toks, partial, reversed=reversed
            ):
                res.append(
                    PartialMatch(
                        name=self.name,
                        start=partial.end,
                        end=match.end,
                        defns=match.defns,
                        data=[self.which_branch(branch_idx), match],
                    )
                )
        return res


class VarDefn(Matcher):
    def __init__(self, var_name, child_matcher, name=None):
        super().__init__()
        self.name = f"VarDefn:{var_name}:{randstr()}" if name is None else name
        self.var_name = var_name
        self.child_matcher = child_matcher

    def get_partial_match_extensions(self, toks, partial, reversed=False):
        child_matches = self.child_matcher.get_partial_match_extensions(
            toks, partial, reversed=reversed
        )

        res = []
        for match in child_matches:
            res.append(
                PartialMatch(
                    name=self.name,
                    start=match.start,
                    end=match.end,
                    defns=match.defns | {self.var_name: toks[match.start : match.end]},
                    data=match,
                )
            )

        return res

    def extra_repr(self):
        return f"(var_name): '{self.var_name}'"


class VarRef(Matcher):
    def __init__(self, var_name, name=None):
        super().__init__()
        self.name = f"VarRef({var_name}):{randstr()}" if name is None else name
        self.var_name = var_name

    def get_partial_match_extensions(self, toks, partial, reversed):
        # unaffected by reversed

        if self.var_name not in partial.defns:
            return []
        var_defn = partial.defns[self.var_name]

        if toks_eq(toks[partial.end : partial.end + len(var_defn)], var_defn):
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + len(var_defn),
                    defns=partial.defns,
                    data=None,
                )
            ]
        else:
            return []

    def extra_repr(self):
        return f"'{self.var_name}'"


class Lookahead(Matcher):
    def __init__(self, child_module, is_neg: bool, name=None):
        super().__init__()
        self.name = f"Lookaround:{randstr()}" if name is None else name
        self.child_module = child_module
        self.is_neg = is_neg

    def get_partial_match_extensions(self, toks, partial, reversed):

        matches = self.child_module.get_partial_match_extensions(
            toks, partial, reversed=False
        )
        if self.is_neg:
            if matches:
                return []
            else:
                return [
                    PartialMatch(
                        name=self.name,
                        start=partial.end,
                        end=partial.end,
                        defns=partial.defns,
                        data=None,
                    )
                ]
        else:
            matches = [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end,
                    defns=match.defns,
                    data=match,
                )
                for match in matches
            ]
        return matches

    def extra_repr(self):
        return f"(is_neg): {self.is_neg}"


class Lookbehind(Matcher):
    def __init__(self, child_module, is_neg: bool, name=None):
        super().__init__()
        self.name = f"Lookaround:{randstr()}" if name is None else name
        self.child_module = child_module
        self.is_neg = is_neg

    def get_partial_match_extensions(self, toks, partial, reversed):
        reversed_toks = toks[::-1]
        reversed_partial = PartialMatch(
            name=partial.name,
            start=len(toks) - partial.end,
            end=len(toks) - partial.end,
            defns=frozendict({k: v[::-1] for k, v in partial.defns.items()}),
            data=partial.data,
        )

        matches = self.child_module.get_partial_match_extensions(
            reversed_toks, reversed_partial, reversed=True
        )
        if self.is_neg:
            if len(matches) > 0:
                return []
            else:
                return [
                    PartialMatch(
                        name=self.name,
                        start=partial.end,
                        end=partial.end,
                        defns=partial.defns,
                        data=None,
                    )
                ]
        else:
            matches = [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end,
                    defns={k: v[::-1] for k, v in match.defns.items()},
                    data=match,
                )
                for match in matches
            ]
        return matches

    def extra_repr(self):
        return f"(is_neg): {self.is_neg}"


class LearnedConst(Matcher):
    def __init__(self, child_module, name=None):
        super().__init__()
        self.name = f"LearnedConst:{randstr()}" if name is None else name
        self.child_module = child_module
        self.bias = Embed(1)

    def get_partial_match_extensions(self, toks, partial, reversed):
        matches = self.child_module.get_partial_match_extensions(
            toks, partial, reversed
        )
        matches = [
            PartialMatch(
                name=self.name,
                start=match.start,
                end=match.end,
                defns=match.defns,
                data=self.bias(0),
            )
            for match in matches
        ]
        return matches
