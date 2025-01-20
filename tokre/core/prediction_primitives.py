from __future__ import annotations
import torch
import torch.nn as nn

from typing import Union, Optional
from dataclasses import dataclass


def randstr(length=6):
    import random
    import string

    return "".join(random.choices(string.ascii_uppercase, k=length))


"""
Every tokre match comes with PredData. This is optionally-nested data that represents the branches a match or partial match took.
The idea is that we want to incorporate *how* the match occured into our prediction of a feature.

For example, the script ( a| an) could be matched in two ways: via the token ` a` or via the token ` an`.
This would be represented inside an EmbedData dataclass with data=0 or data=1.

The simplest PredData is None representing no data.
The next step up in complexity is is EmbedData, which contains an int or tuple of ints.
A PartialMatch itself can be PredData
Finally, a list of PredData can be PredData.
"""


@dataclass
class EmbedData:
    name: str
    data: Union[int, tuple[int]]


PredData = Union["PartialMatch", EmbedData, None, list["PredData"]]


@dataclass
class PartialMatch:
    """PartialMatch represents a partial match for some token sequence.

    Attributes:
        start: Starting index of the partial match
        end: Ending index of the partial match
        defns: Variable definitions created by [<var_ref>=(...)] syntax
        data: Existing match data used for prediction
    """

    name: str
    start: int
    end: int
    defns: frozendict
    data: PredData

    def __len__(self):
        return self.end - self.start


def is_pred_data(obj):
    return (
        isinstance(obj, (PartialMatch, EmbedData))
        or obj is None
        or (isinstance(obj, list) and all(is_pred_data(item) for item in obj))
    )


"""
Here we define the core primitive modules that take in PredData to produce a single predicted activation number: Embed and Mixer.
    Embed contains an n-d PyTorch parameter that's indexed into to produce a single prediction float.
    Mixer combines outputs from Embed/Mixers into a single prediction float.

The 'matcher' modules in modules.py use these to 
"""


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
            self.bilinear_pre_bias = nn.Parameter(
                torch.zeros(
                    d_module + 1,
                )
            )
            self.bilinear_param = nn.Parameter(torch.zeros(d_module + 1, d_module + 1))

        if self.linear:
            self.linear_pre_bias = nn.Parameter(
                torch.zeros(
                    d_module + 1,
                )
            )
            self.linear_param = nn.Parameter(torch.ones(d_module + 1) / d_module)

        self.bias = nn.Parameter(torch.zeros(1)[0])

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
        y = self.bias
        if self.bilinear:
            pre_bilinear = x  # + self.bilinear_pre_bias[:D]
            y = y + torch.einsum(
                "i, ij, j", pre_bilinear, self.bilinear_param[:D, :D], pre_bilinear
            )
        if self.linear:
            pre_linear = x  # + self.linear_pre_bias[:D]
            y = y + self.linear_param[:D] @ pre_linear
        return y

    def __repr__(self):
        return f"Mixer({self.d_module}, bilinear={self.bilinear}, linear={self.linear})"
