"""Major-complex search over MultivaluedNetwork (exact IIT-4.0 Φ)."""

from __future__ import annotations

from typing import Sequence

from pyphi import exceptions, utils
from pyphi.new_big_phi import NullPhiStructure, SystemIrreducibilityAnalysis

from .network import MultivaluedNetwork
from .sia_mv import sia
from .subsystem_mv import MultivaluedSubsystem


def maximal_complex(
    network: MultivaluedNetwork,
    state: Sequence[int],
) -> SystemIrreducibilityAnalysis | NullPhiStructure:
    """Return the max-Φ complex at ``state`` (exact, mixed radix).

    Same selection as stock ``new_big_phi.maximal_complex``: subsystems are
    visited in ``reachable_subsystems`` order (``utils.powerset(...,
    reverse=True)``: largest first, reverse lexicographic within a size),
    reducible ones are dropped, and ``max`` picks the largest Φ. Φ values are
    ``PyPhiFloat``, so a later candidate only displaces an earlier one when it
    is larger by more than ``10 ** -config.PRECISION``; otherwise the earlier
    one stands, as in stock.
    """

    def irreducible_complexes():
        for subset in utils.powerset(network.node_indices, nonempty=True, reverse=True):
            try:
                sub = MultivaluedSubsystem(network, state, nodes=subset)
            except exceptions.StateUnreachableError:
                continue
            result = sia(sub)
            if result:
                yield result

    return max(irreducible_complexes(), default=NullPhiStructure())
