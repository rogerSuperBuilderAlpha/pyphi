"""Major-complex search over MultivaluedNetwork (exact IIT-4.0 Φ)."""

from __future__ import annotations

from typing import Optional, Sequence

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

    Subsystems are visited in the order stock ``reachable_subsystems`` uses
    (``utils.powerset(..., reverse=True)``: largest first, and within a size
    in reverse lexicographic order), and the first subsystem that reaches the
    maximum Φ wins, as with stock ``maximal_complex``'s ``max``. Ties
    therefore resolve to the same complex as stock.
    """
    best: Optional[SystemIrreducibilityAnalysis] = None
    for subset in utils.powerset(network.node_indices, nonempty=True, reverse=True):
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
