from __future__ import annotations
import torch
import torch.nn as nn

from typing import Union, TypeAlias, Optional
from dataclasses import dataclass
from types import MappingProxyType


def randstr(length=4):
    import random
    import string

    return "".join(random.choices(string.ascii_uppercase, k=length))


@dataclass
class EmbedData:
    name: str
    data: Union[int, tuple[int]]


class Embed(nn.Module):
    def __init__(
        self, embed_shape: Union[list[int], tuple[int], int], name: Optional[str] = None
    ):
        super().__init__()

        self.name = "Embed:" + randstr() if name is None else name

        if isinstance(embed_shape, int):
            embed_shape = (embed_shape,)

        embed = torch.ones(embed_shape)

        self.embed_shape = embed_shape
        self.embed = nn.Parameter(embed)

    def __call__(self, *indices):
        for i, index in enumerate(indices):
            assert index >= 0, "index must be >= 0"
            assert index < self.embed_shape[i], "index too large"

        if len(indices) == 1:
            indices = indices[0]

        return EmbedData(name=self.name, data=indices)
    
    def pred(self, embed_data: EmbedData):
        indices = tuple(embed_data.data)
        assert all([isinstance(i, int) for i in indices])
        assert len(indices) == len(self.embed_shape)

        assert all([(idx >= 0 and idx < embed_max) for (idx, embed_max) in zip(indices, self.embed_shape)])

        return self.embed[tuple(embed_data.data)]
        

    def __repr__(self):
        return f"Embed{self.embed_shape}"


class Mixer(nn.Module):
    def __init__(self, d_module: int, bilinear=False, linear=False):
        super().__init__()
        assert bilinear is True or linear is True  # or d_module == 1
        assert d_module > 0

        self.d_module = d_module

        self.bilinear = bilinear
        self.linear = linear

        if self.bilinear:
            self.bilinear_pre_bias = nn.Parameter(torch.zeros(d_module+1,))
            self.bilinear_param = nn.Parameter(torch.zeros(d_module+1, d_module+1))

        if self.linear:
            self.linear_pre_bias = nn.Parameter(torch.zeros(d_module+1,))
            self.linear_param = nn.Parameter(torch.ones(d_module+1) / d_module)
        
        # self.inp_bias = nn.Parameter(torch.ones(d_module + 1) / d_module)

    def device(self):
        if hasattr(self, "bilinear_param"):
            return self.bilinear_param.device
        elif hasattr(self, "linear_param"):
            return self.linear_param.device
        else:
            raise ValueError(
                "Mixer object has neither self.bilinear_param or self.linear_param, which isn't expected."
            )

    def dtype(self):
        if hasattr(self, "bilinear_param"):
            return self.bilinear_param.dtype
        elif hasattr(self, "linear_param"):
            return self.linear_param.dtype
        else:
            raise ValueError(
                "Mixer object has neither self.bilinear_param or self.linear_param, which isn't expected."
            )
        
    def forward(self, x):
        D = x.shape[0]
        y = torch.tensor(0.)
        if self.bilinear:
            pre_bilinear = x + self.bilinear_pre_bias[:D]
            y = y + torch.einsum('i, ij, j', pre_bilinear, self.bilinear_param[:D, :D], pre_bilinear)
        if self.linear:
            pre_linear = x + self.linear_pre_bias[:D]
            y = y + self.linear_param[:D] @ pre_linear
        return y






    # def forward(self, x):
    #     assert len(x.shape) == 2, "input must be batched vectors"

    #     if self.linear and not self.bilinear:
    #         return x @ self.linear_param
    #     elif self.bilinear and not self.linear:
    #         return torch.einsum("mn, bn, bm -> b", self.bilinear_param, x, x)
    #     elif self.linear and self.bilinear:
    #         return (x @ self.linear_param) + torch.einsum(
    #             "mn, bn, bm -> b", self.bilinear_param, x, x
    #         )
    #     else:
    #         assert self.d_module == 1
    #         return x[..., 0]

    # def pred(self, data: list[PredData]):


    def __repr__(self):
        return f"Mixer({self.d_module}, bilinear={self.bilinear}, linear={self.linear})"




# PredData: TypeAlias = Union["PartialMatch", EmbedData, None, list["PredData"]]
PredData = Union["PartialMatch", EmbedData, None, list["PredData"]]


@dataclass
class PartialMatch:
    name: str
    start: int
    end: int
    defns: MappingProxyType
    data: PredData

    def __len__(self):
        return self.end - self.start


def is_pred_data(obj):
    return (
        isinstance(obj, (PartialMatch, EmbedData, list)) or
        obj is None or
        (isinstance(obj, list) and all(is_pred_data(item) for item in obj))
    )
    


def batched_extend_matches(
    toks, partial_matches: list[PartialMatch], matcher, reversed
):
    """
    matcher is a Matcher that implements matches
    """
    new_partials = []

    for partial in partial_matches:
        assert isinstance(
            partial.data, list
        ), f"Provided partial match group_data must be represented as a list: {partial.data=}"
        matcher_results = matcher.matches(toks, partial, reversed=reversed)

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


def pyregex_literal(toks):
    # [STUB]
    return "".join(toks)


def toks_eq(toks_a: list[str], toks_b: list[str]):
    return (len(toks_a) == len(toks_b)) and all(
        [a == b for (a, b) in zip(toks_a, toks_b)]
    )


class Toks(nn.Module):
    def __init__(self, toks: list[str], name: Optional[str] = None):
        super().__init__()
        self.name = "Toks:" + randstr() if name is None else name
        self.toks = toks

    def matches(self, toks, partial, reversed: bool):
        if toks_eq(toks[partial.end : partial.end + len(self.toks)], self.toks):
            match = PartialMatch(
                name=self.name,
                start=partial.end,
                end=partial.end + len(self.toks),
                defns=partial.defns,
                data=None,
            )
            return [match]
        else:
            return []

    @property
    def pyregex(self):
        return pyregex_literal(self.toks)

    def __repr__(self):
        return f"Toks({self.toks})"


Inf = float("inf")


class Repeat(nn.Module):
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

    def matches(self, toks, partial, reversed):
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

    @property
    def pyregex(self):
        pyregex_max = "" if self.max == Inf else str(self.max)
        return (
            f"({self.child_matcher.pyregex})"
            + r"{"
            + str(self.min)
            + ","
            + pyregex_max
            + r"}"
        )


class Phrase(nn.Module):
    def __init__(self, matchers, name=None):
        super().__init__()
        self.name = "Phrase:" + randstr() if name is None else name
        self.matchers = nn.ModuleList(matchers)

        self.mixer = Mixer(len(self.matchers), linear=True, bilinear=True)

    def matches(self, toks, partial, reversed: bool):
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

    @property
    def pyregex(self):
        return f"({''.join([child.pyregex for child in self.matchers])})"


class Wildcard(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = "Wildcard:" + randstr() if name is None else name

    def matches(self, toks, partial, reversed: bool):
        if partial.end < len(toks):
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + 1,
                    defns=partial.defns,
                    data=partial.data,
                )
            ]
        else:
            return []

    @property
    def pyregex(self):
        return f"."


class OrGroup(nn.Module):
    def __init__(self, matchers, name=None):
        super().__init__()
        self.name = "OrGroup:" + randstr() if name is None else name
        self.branches = nn.ModuleList(matchers)

        self.which_branch = Embed(len(self.branches))

        self.mixer = Mixer(2, bilinear=True)

    def matches(self, toks, partial, reversed=False):
        res = []
        for branch_idx, branch in enumerate(self.branches):
            for match in branch.matches(toks, partial, reversed=reversed):
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

    @property
    def pyregex(self):
        return f"({'|'.join([child.pyregex for child in self.matchers])})"


class VarDefn(nn.Module):
    def __init__(self, var_name, child_matcher, name=None):
        super().__init__()
        self.name = f"VarDefn:{var_name}:{randstr()}" if name is None else name
        self.var_name = var_name
        self.child_matcher = child_matcher

    def matches(self, toks, partial, reversed=False):
        child_matches = self.child_matcher.matches(toks, partial, reversed=reversed)

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

    @property
    def pyregex(self):
        f"(?P<{self.var_name}>{self.child_matcher.pyregex})"

    def extra_repr(self):
        return f"(var_name): '{self.var_name}'"


class VarRef(nn.Module):
    def __init__(self, var_name, name=None):
        super().__init__()
        self.name = f"VarRef({var_name}):{randstr()}" if name is None else name
        self.var_name = var_name

    def matches(self, toks, partial, reversed):
        # unaffected by reversed

        if self.var_name not in partial.defns:
            return []
        var_defn = partial.defns[self.var_name]

        if toks_eq(toks[partial.end : partial.end + len(var_defn)], var_defn):
            return [
                PartialMatch(
                    name=self.name,
                    start=partial.end,
                    end=partial.end + len(partial.defns),
                    defns=partial.defns,
                    data=None,
                )
            ]
        else:
            return []

    def extra_repr(self):
        return f"'{self.var_name}'"


class LearnedConst(nn.Module):
    def __init__(self, child_module, name=None):
        super().__init__()
        self.name = f"LearnedConst:{randstr()}" if name is None else name
        self.child_module = child_module
        self.bias = Embed(1)

    def matches(self, toks, partial, reversed):
        matches = self.child_module.matches(toks, partial, reversed)
        matches = [
            PartialMatch(
                name=self.name,
                start=match.start,
                end=match.end,
                defns=match.defns,
                data=self.bias(0)
            )
            for match in matches
        ]
        return matches
    
    @property
    def pyregex(self):
        return self.child_module.pyregex