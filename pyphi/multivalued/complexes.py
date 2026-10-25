"""Major-complex search over MultivaluedNetwork (exact IIT-4.0 Φ)."""

from __future__ import annotations

from itertools import chain, combinations
from typing import Optional, Sequence

from pyphi import exceptions
from pyphi.new_big_phi import NullPhiStructure, SystemIrreducibilityAnalysis

from .network import MultivaluedNetwork
from .sia_mv import sia
from .subsystem_mv import MultivaluedSubsystem


def _powerset(indices):
    # Largest first so Φ ties prefer the bigger complex (matches stock
    # maximal_complex when full system is yielded before subsets).
    return chain.from_iterable(
        combinations(indices, r) for r in range(len(indices), 0, -1)
    )


def maximal_complex(
    network: MultivaluedNetwork,
    state: Sequence[int],
) -> SystemIrreducibilityAnalysis | NullPhiStructure:
    """Return the max-Φ complex at ``state`` (exact, mixed radix)."""
    best: Optional[SystemIrreducibilityAnalysis] = None
    for subset in _powerset(network.node_indices):
        try:
            sub = MultivaluedSubsystem(network, state, nodes=subset)
        except exceptions.StateUnreachableError:
            continue
        result = sia(sub)
        if not result:
            continue
        if best is None or float(result.phi) > float(best.phi):
            best = result
    if best is None:
        return NullPhiStructure()
    return best
