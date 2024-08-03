from tokre.core.modules import Embed, Mixer, VarRef, PartialMatch, randstr, toks_eq
from torch import nn
import tiny_model


def tok_split(s):
    tok_ids = tiny_model.enc(s)[0]
    return tiny_model.raw_toks[tok_ids.tolist()].tolist()


def get_literal_variants(tok_literal: list[str]):
    literal_str = ("".join(tok_literal)).strip()
    variants = [tok_split(literal_str), tok_split(" " + literal_str)]
    return variants


class VarPrefix(nn.Module):
    def __init__(self, var_ref, max_len=10):
        super().__init__()
        assert isinstance(var_ref, VarRef), var_ref
        self.name = f"VarPrefix:{var_ref}:{randstr()}"
        self.var_name = var_ref.var_name
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


DEFINED_MACROS = {"var_prefix": VarPrefix}
