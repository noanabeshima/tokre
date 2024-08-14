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

import ray
from tqdm import tqdm

ray.init(ignore_reinit_error=True)


def collect_matches(module, toks, aggr="longest"):
    assert aggr in ["shortest", "longest"]
    starting_matches = [
        PartialMatch(
            name="start",
            start=start_idx,
            end=start_idx,
            defns=frozendict(),
            data=None,
        )
        for start_idx in range(len(toks))
    ]

    unaggregated_matches = [
        match
        for start_match in starting_matches
        for match in module.matches(toks, start_match, reversed=False)
    ]

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
    return isinstance(data, int) or (
        isinstance(data, tuple) and all([isinstance(x, int) for x in data])
    )


from tokre.core.modules import is_pred_data, Embed


def pred(module, match_data):
    assert is_pred_data(match_data) or (
        isinstance(module, Embed)
        and isinstance(match_data, int)
        or isinstance(match_data, tuple)
    )
    assert hasattr(module, "name")

    if isinstance(module, Embed):
        assert (
            isinstance(match_data, int)
            or isinstance(match_data, tuple)
            and all([isinstance])
        )
        return module.embed[match_data]

    if isinstance(match_data, list):
        assert hasattr(module, "mixer"), module
        preds = [torch.tensor(1.0)] + [pred(module, data) for data in match_data]
        preds = torch.stack(preds)

        return module.mixer(preds)

    elif isinstance(match_data, PartialMatch) or isinstance(match_data, EmbedData):
        match = match_data

        assert hasattr(module, "name_to_submodule")
        assert match.name in module.name_to_submodule, (module.name, match.name)
        return pred(module.name_to_submodule[match.name], match.data)

    elif match_data is None:
        return torch.tensor(1.0)

    else:
        raise ValueError("Unexpected match_data", match_data)


@ray.remote(num_cpus=1)
class ParallelModule:
    def __init__(self, module, aggr):
        self.module = module
        self.aggr = aggr

    def get_matches(self, docs):
        assert not isinstance(docs[0], str) and isinstance(docs[0], Iterable)
        matches = [
            collect_matches(self.module, toks=doc, aggr=self.aggr) for doc in docs
        ]
        return matches


class SynthFeat(nn.Module):
    def __init__(self, script, aggr="longest", n_actors=10, batch_size=100):
        super().__init__()
        assert aggr in ["longest", "shortest"]
        self.module = tokre.compile(script)
        self.aggr = aggr
        self.optimizer = AdamWScheduleFree(self.module.parameters(), lr=1e-3)

        self.parallel_matchers = [
            ParallelModule.remote(self.module, self.aggr) for _ in range(n_actors)
        ]
        self.batch_size = batch_size

    def get_matches(self, toks: list[str]):
        if isinstance(toks[0], list) or (
            isinstance(toks, np.ndarray) and len(toks.shape) == 2
        ):
            docs = toks
            batched_docs = [
                docs[i : i + self.batch_size]
                for i in range(0, len(docs), self.batch_size)
            ]
            batched_matches = ray.get(
                [
                    self.parallel_matchers[
                        i % len(self.parallel_matchers)
                    ].get_matches.remote(batch)
                    for i, batch in enumerate(batched_docs)
                ]
            )
            per_doc_matches = [match for batch in batched_matches for match in batch]
            return per_doc_matches

        matches = collect_matches(self.module, toks=toks, aggr=self.aggr)
        return matches

    def get_mask(self, toks: Iterable):
        if isinstance(toks[0], Iterable) and not isinstance(toks[0], str):
            assert all([isinstance(tok, str) for tok in toks[0]])
            matches = self.get_matches(toks)
            mask = torch.zeros((len(toks), len(toks[0])))
            for doc_idx, doc_matches in enumerate(matches):
                for match in doc_matches:
                    mask[doc_idx, match.end - 1] = 1.0
            return mask
        else:
            matches = self.get_matches(toks)
            mask = torch.zeros(len(toks))
            for match in matches:
                mask[match.end - 1] = 1.0

        return mask

    @torch.no_grad()
    def get_acts(self, toks):
        if isinstance(toks, Iterable) and isinstance(toks[0], str):
            synth_acts = torch.zeros(len(toks))
            matches = self.get_matches(toks)
            for match in matches:
                prediction = pred(self.module, match.data)
                synth_acts[match.end - 1] = prediction
        else:
            assert isinstance(toks, Iterable)
            assert isinstance(toks[0], Iterable)
            # return torch.stack([self.get_acts(doc) for doc in toks], dim=0)
            synth_acts = torch.zeros((len(toks), len(toks[0])))
            doc_matches = self.get_matches(toks)
            for doc_idx, matches in enumerate(doc_matches):
                for match in matches:
                    with torch.no_grad():
                        prediction = pred(self.module, match.data)
                        synth_acts[doc_idx, match.end-1] = prediction
        
        return synth_acts


    def train(self, toks, acts):
        print("getting matches")
        all_matches = self.get_matches(toks)
        print("training")
        for doc_matches, doc_acts in tqdm(zip(all_matches, acts)):
            for match in doc_matches:
                act = doc_acts[match.end - 1]
                loss = (pred(self.module, match.data) - act).abs()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @property
    def pyregex(self):
        return self.module.pyregex
