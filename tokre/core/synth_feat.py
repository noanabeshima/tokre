import torch.nn as nn
import tokre
from tokre.core.modules import PartialMatch
from types import MappingProxyType
from tokre.core.parsing import parse
from tokre.core.tree_to_module import compile

from schedulefree import AdamWScheduleFree
import numpy as np
import torch

def collect_matches(matcher, toks, aggr='longest'):
    assert aggr in ['shortest', 'longest']
    starting_matches = [
            PartialMatch(
                name='start',
                start=start_idx,
                end=start_idx,
                defns=MappingProxyType({}),
                data=None,
            )
            for start_idx in range(len(toks))
        ]

    unaggregated_matches = [match for start_match in starting_matches for match in matcher.matches(toks, start_match, reversed=False)]
    
    end_to_aggr_match = {}
    for match in unaggregated_matches:
        if match.end in end_to_aggr_match:
            aggr_match = end_to_aggr_match[match.end]
            if len(match) > len(aggr_match):
                end_to_aggr_match[match.end] = match
        else:
            end_to_aggr_match[match.end] = match
    
    return list(end_to_aggr_match.values())


class SynthFeat(nn.Module):
    def __init__(self, script, aggr='longest'):
        super().__init__()
        assert aggr in ['longest', 'shortest']
        self.module = tokre.compile(script)
        self.aggr=aggr
        self.optimizer = AdamWScheduleFree(self.module.parameters())
        
    def get_matches(self, toks: list[str]):
        matches = collect_matches(self.module, toks=toks, aggr=self.aggr)
        return matches
        
    def get_act_mask(self, toks: list):
        if isinstance(toks[0], list):
            assert all([isinstance(tok, str) for tok in toks[0]])
            mask = torch.stack([self.get_act_mask(row) for row in toks], dim=0)
        else:
            matches = self.get_matches(toks)
            mask = torch.zeros(len(toks))
            for match in matches:
                mask[match.end-1] = 1.
        
        return mask

    # def get_pred(self, match):

