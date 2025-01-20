import torch.nn as nn
import tokre
from tokre.core.matchers import PartialMatch, EmbedData, Matcher, is_pred_data, Embed
from tokre.core.parsing import parse
from tokre.core.tree_to_matcher import script_to_matcher

from schedulefree import AdamWScheduleFree
import numpy as np
import torch
from frozendict import frozendict

from typing import Iterable
from tqdm import tqdm


def collect_matches(
    matcher: Matcher, toks: Iterable[str], aggr="longest"
) -> list[PartialMatch]:
    """
    Given a matcher defined in tokre.core.matchers.py or tokre.core.macros.py, which implements the method get_partial_match_extensions,
        - Calculate all matches
        - For each possible match-end, choose the (longest|shortest) match with that end index
            (no guarantees if multiple matches have the same extremal length)
        - Return these matches
    """
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

    # For every possible starting point, get all matches from that point.
    unaggregated_matches = [
        match
        for start_match in starting_matches
        for match in matcher.get_partial_match_extensions(
            toks, start_match, reversed=False
        )
    ]

    # dictionary from end of a match (an int) to the longest match seen so far that ends there (PartialMatch)
    end_to_aggr_match = {}

    # If two matches end at the same point, keep just the longest match.
    # No guarantees are provided on which match will be selected if two matches have the same end point and the same length.
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


def get_single_prediction(matcher, match_data):
    assert is_pred_data(match_data) or (
        isinstance(matcher, Embed)
        and isinstance(match_data, int)
        or isinstance(match_data, tuple)
    )
    assert hasattr(matcher, "name")

    if isinstance(matcher, Embed):
        assert (
            isinstance(match_data, int)
            or isinstance(match_data, tuple)
            and all([isinstance(d, int) for i in match_data])
        )
        return matcher.embed[match_data]

    if isinstance(match_data, list):
        assert hasattr(matcher, "mixer"), matcher
        preds = [torch.tensor(1.0)] + [
            get_single_prediction(matcher, data) for data in match_data
        ]
        preds = torch.stack(preds)

        return matcher.mixer(preds)

    elif isinstance(match_data, PartialMatch) or isinstance(match_data, EmbedData):
        match = match_data

        assert hasattr(matcher, "name_to_child_matcher")
        assert match.name in matcher.name_to_child_matcher, (matcher.name, match.name)
        return get_single_prediction(
            matcher.name_to_child_matcher[match.name], match.data
        )

    elif match_data is None:
        return torch.tensor(1.0)

    else:
        raise ValueError("Unexpected match_data", match_data)


def tok_ids_to_string(toks):
    """If toks is a tok id array, convert to a numpy array of token strings"""
    if isinstance(toks, np.ndarray) or isinstance(toks, torch.Tensor):
        vocab = tokre.get_vocab()
        return vocab[toks]
    else:
        return toks


class SynthFeat(nn.Module):
    def __init__(self, script, aggr="longest", batch_size=100):
        super().__init__()
        assert aggr in ["longest", "shortest"]
        self.matcher = tokre.script_to_matcher(script)
        self.aggr = aggr
        self.optimizer = AdamWScheduleFree(self.matcher.parameters(), lr=1e-3)

        self.batch_size = batch_size

    def get_matches(self, toks):
        if isinstance(toks[0], list) or (
            isinstance(toks, np.ndarray) and len(toks.shape) == 2
        ):
            # In this branch, toks is a list of documents (toks: list[list[str]] or 2d numpy array of strs)
            docs = toks

            pbar = tqdm(docs, desc="collecting matches")
            per_doc_matches = [
                collect_matches(self.matcher, toks=doc, aggr=self.aggr) for doc in pbar
            ]
            return per_doc_matches

        matches = collect_matches(self.matcher, toks=toks, aggr=self.aggr)
        return matches

    def get_mask(self, toks: Iterable):
        """
        Get mask of token positions where at least one match ends (i.e. there's a match there)

        For example if the SynthFeat script is
        ` a ( gorgeous| beautiful| silly) frog`

        if the input is
        np.array([
            [" She", " saw", " a", " silly", " frog", " there"],
            [" What", " did", " Ada", " see", " then", "?"]
        ])

        this method would return

        torch.tensor([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        """
        toks = tok_ids_to_string(toks)

        if isinstance(toks[0], Iterable) and not isinstance(toks[0], str):
            # toks is a list of documents
            assert all([isinstance(tok, str) for tok in toks[0]])
            matches = self.get_matches(toks)
            mask = torch.zeros((len(toks), len(toks[0])), dtype=int)
            for doc_idx, doc_matches in enumerate(matches):
                for match in doc_matches:
                    mask[doc_idx, match.end - 1] = 1
            return mask
        else:
            matches = self.get_matches(toks)
            mask = torch.zeros(len(toks), dtype=int)
            for match in matches:
                mask[match.end - 1] = 1

        return mask

    @torch.no_grad()
    def get_preds(self, toks, n_matchers=1):
        """
        Get feature predictions.

        Used after training this SynthFeat with the `train` method.
        """
        toks = tok_ids_to_string(toks)

        if isinstance(toks, Iterable) and isinstance(toks[0], str):
            synth_acts = torch.zeros(len(toks))
            matches = self.get_matches(toks, n_matchers=n_matchers)
            for match in matches:
                prediction = get_single_prediction(self.matcher, match.data)
                synth_acts[match.end - 1] = prediction
        else:
            assert isinstance(toks, Iterable)
            assert isinstance(toks[0], Iterable)
            synth_acts = torch.zeros((len(toks), len(toks[0])))
            doc_matches = self.get_matches(toks, n_matchers=n_matchers)
            for doc_idx, matches in enumerate(doc_matches):
                for match in matches:
                    with torch.no_grad():
                        prediction = get_single_prediction(self.matcher, match.data)
                        synth_acts[doc_idx, match.end - 1] = prediction

        return synth_acts

    def train(self, toks, acts, parallel=True):
        print("getting matches")
        all_matches = self.get_matches(toks, parallel=parallel)
        print("training")
        for doc_matches, doc_acts in tqdm(zip(all_matches, acts)):
            for match in doc_matches:
                act = doc_acts[match.end - 1]

                loss = (get_single_prediction(self.matcher, match.data) - act) ** 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def compile(script: str):
    return SynthFeat(script)
