import torch.nn as nn
import tokre
from tokre.core.modules import PartialMatch, EmbedData
from tokre.core.parsing import parse
from tokre.core.tree_to_module import compile

from schedulefree import AdamWScheduleFree
import numpy as np
import torch
from frozendict import frozendict

from typing import Iterable

def collect_matches(matcher, toks, aggr='longest'):
    assert aggr in ['shortest', 'longest']
    starting_matches = [
            PartialMatch(
                name='start',
                start=start_idx,
                end=start_idx,
                defns=frozendict(),
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

def is_int_or_tuple_of_ints(data):
    return isinstance(data, int) or (isinstance(data, tuple) and all([isinstance(x, int) for x in data]))

from tokre.core.modules import is_pred_data, Embed


def pred(module, match_data):
    assert (is_pred_data(match_data) or (isinstance(module, Embed) and isinstance(match_data, int) or isinstance(match_data, tuple)))
    assert hasattr(module, 'name')

    if isinstance(module, Embed):
        assert isinstance(match_data, int) or isinstance(match_data, tuple) and all([isinstance])
        return (module.embed[match_data])


    if isinstance(match_data, list):
        assert hasattr(module, 'mixer')
        preds =  [torch.tensor(1.)] + [pred(module, data) for data in match_data]
        preds = torch.stack(preds)
        
        return module.mixer(preds)

    elif isinstance(match_data, PartialMatch) or isinstance(match_data, EmbedData):
        match = match_data

        assert hasattr(module, 'name_to_submodule')
        assert match.name in module.name_to_submodule, (match.name, module.name)
        return pred(module.name_to_submodule[match.name], match.data)
    
    elif match_data is None:
        return torch.tensor(1.)
    
    else:
        raise ValueError('Unexpected match_data', match_data)


class SynthFeat(nn.Module):
    def __init__(self, script, aggr='longest'):
        super().__init__()
        assert aggr in ['longest', 'shortest']
        self.module = tokre.compile(script)
        self.aggr=aggr
        self.optimizer = AdamWScheduleFree(self.module.parameters(), lr=1e-2)
        
    def get_matches(self, toks: list[str]):
        matches = collect_matches(self.module, toks=toks, aggr=self.aggr)
        return matches
        
    def get_act_mask(self, toks: Iterable):
        if isinstance(toks[0], Iterable) and not isinstance(toks[0], str):
            assert all([isinstance(tok, str) for tok in toks[0]])
            mask = torch.stack([self.get_act_mask(row) for row in toks], dim=0)
        else:
            matches = self.get_matches(toks)
            mask = torch.zeros(len(toks))
            for match in matches:
                mask[match.end-1] = 1.
        
        return mask

    @torch.no_grad()
    def get_acts(self, toks):
        if (isinstance(toks, Iterable) and isinstance(toks[0], str)):
            synth_acts = torch.zeros(len(toks))
            matches = self.get_matches(toks)
            for match in matches:
                prediction = pred(self.module, match.data)
                synth_acts[match.end-1] = prediction
            return synth_acts
        else:
            assert isinstance(toks, Iterable)
            assert isinstance(toks[0], Iterable)
            return torch.stack([self.get_acts(doc) for doc in toks], dim=0)

    def train(self, toks, acts):
        for doc, doc_acts in zip(toks, acts):
            synth_matches = self.get_matches(doc)
            for match in synth_matches:
                act = doc_acts[match.end-1]
                loss = ((pred(self.module, match.data)-act)**2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    @property
    def pyregex(self):
        return self.module.pyregex