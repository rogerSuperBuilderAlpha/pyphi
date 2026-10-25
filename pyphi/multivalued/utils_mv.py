# -*- coding: utf-8 -*-
"""Mixed-radix state helpers (no binary log2 assumptions)."""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np

from .conditional_independence import decode_state


def all_states_mixed(
    num_states_per_node: Sequence[int],
) -> Iterable[tuple[int, ...]]:
    """Yield all little-endian mixed-radix states (node 0 varies fastest)."""
    if not num_states_per_node:
        yield ()
        return
    # product iterates last axis fastest; we want node 0 fastest → reverse
    ranges = [range(k) for k in num_states_per_node]
    for st in product(*ranges):
        yield st


def repertoire_shape_mv(
    all_node_indices: Sequence[int],
    purview: Sequence[int],
    num_states_per_node: Sequence[int],
) -> list[int]:
    """Repertoire shape: alphabet size on purview axes, 1 elsewhere."""
    return [
        int(num_states_per_node[i]) if i in purview else 1 for i in all_node_indices
    ]


def max_entropy_mv(
    all_node_indices: Sequence[int],
    purview: Sequence[int],
    num_states_per_node: Sequence[int],
) -> np.ndarray:
    shape = repertoire_shape_mv(all_node_indices, purview, num_states_per_node)
    n = int(np.prod([num_states_per_node[i] for i in purview])) if purview else 1
    return np.ones(shape, dtype=float) / float(n)


def state_of(nodes: Sequence[int], network_state: Sequence[int]) -> tuple[int, ...]:
    return tuple(network_state[n] for n in nodes) if nodes else ()


def factorize_node_tpms(
    sbs: np.ndarray, num_states_per_node: Sequence[int]
) -> list[np.ndarray]:
    """Per-node categorical effect TPMs, shape ``(*ks, k_i)``.

    ``tpm[i][prev_state + (s,)] = P(X_i'=s | X=prev)``.
    """
    ks = tuple(int(k) for k in num_states_per_node)
    n_states = int(np.prod(ks))
    sbs = np.asarray(sbs, dtype=float)
    if sbs.shape != (n_states, n_states):
        raise ValueError(f"SBS shape {sbs.shape} != {(n_states, n_states)}")

    node_tpms = []
    for i, k_i in enumerate(ks):
        tpm_i = np.zeros(ks + (k_i,), dtype=float)
        for prev_idx in range(n_states):
            prev = decode_state(prev_idx, ks)
            row = sbs[prev_idx]
            for nxt_idx in range(n_states):
                p = row[nxt_idx]
                if p == 0.0:
                    continue
                nxt = decode_state(nxt_idx, ks)
                tpm_i[prev + (nxt[i],)] += p
        node_tpms.append(tpm_i)
    return node_tpms


def cut_node_tpm(
    node_tpm: np.ndarray,
    node_index: int,
    inputs: set[int],
    num_states_per_node: Sequence[int],
) -> np.ndarray:
    """Marginalize non-input parents out of a node TPM (uniform over them)."""
    ks = tuple(int(k) for k in num_states_per_node)
    k_i = ks[node_index]
    out = np.zeros(ks + (k_i,), dtype=float)
    non_inputs = [j for j in range(len(ks)) if j not in inputs]
    # Average over non-input previous dimensions
    if not non_inputs:
        return np.array(node_tpm, copy=True)
    scales = [ks[j] for j in non_inputs]
    denom = float(np.prod(scales))
    # Sum over non-input axes (which are among the first len(ks) axes)
    out = node_tpm.sum(axis=tuple(non_inputs), keepdims=True) / denom
    # Broadcast back to full ks shape for easy indexing
    return np.broadcast_to(out, ks + (k_i,)).copy()
