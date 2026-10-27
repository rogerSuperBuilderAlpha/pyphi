"""Mixed-radix state helpers (no binary log2 assumptions)."""

from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from pyphi.exceptions import StateUnreachableError

from .conditional_independence import decode_state


def all_states_mixed(
    num_states_per_node: Sequence[int],
) -> Iterable[tuple[int, ...]]:
    """Yield all mixed-radix states in little-endian order.

    Node 0 varies fastest, matching ``pyphi.utils.all_states`` and the row
    order of ``encode_state`` / ``decode_state``. Callers that break ties by
    iteration order (``intrinsic_information``) rely on this.
    """
    if not num_states_per_node:
        yield ()
        return
    # ``product`` varies its last factor fastest, so feed it the alphabets in
    # reverse and flip each tuple back.
    ranges = [range(int(k)) for k in reversed(num_states_per_node)]
    for st in product(*ranges):
        yield st[::-1]


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


def condition_node_tpm(
    node_tpm: np.ndarray, condition: Mapping[int, int]
) -> np.ndarray:
    """Fix previous-state axes to the given states (kept as singletons).

    Mirrors ``ExplicitTPM.condition_tpm``: axes that are already singletons
    are left alone, so a TPM that was marginalized or conditioned before can
    be conditioned again.
    """
    index = [slice(None)] * node_tpm.ndim
    for axis, s in condition.items():
        if node_tpm.shape[axis] != 1:
            index[axis] = slice(int(s), int(s) + 1)
    return node_tpm[tuple(index)]


def marginalize_out_mv(node_tpm: np.ndarray, axes: Iterable[int]) -> np.ndarray:
    """Average a node TPM over previous-state axes (kept as singletons).

    Mirrors ``ExplicitTPM.marginalize_out``: the divisor is the actual size
    of each axis, so an axis that is already a singleton (conditioned
    background, or an input marginalized earlier) is left unchanged.
    """
    axes = tuple(sorted(set(int(a) for a in axes)))
    if not axes:
        return np.array(node_tpm, copy=True)
    denom = float(np.prod([node_tpm.shape[a] for a in axes]))
    return node_tpm.sum(axis=axes, keepdims=True) / denom


def cut_node_tpm(
    node_tpm: np.ndarray,
    node_index: int,
    inputs: set[int],
    num_states_per_node: Sequence[int],
) -> np.ndarray:
    """Marginalize non-input parents out of a node TPM (uniform over them).

    The result is broadcast back to the full ``(*ks, k_i)`` shape so callers
    can index it with any complete previous state.
    """
    ks = tuple(int(k) for k in num_states_per_node)
    k_i = ks[node_index]
    non_inputs = [j for j in range(len(ks)) if j not in inputs]
    out = marginalize_out_mv(node_tpm, non_inputs)
    return np.broadcast_to(out, ks + (k_i,)).copy()


def backward_node_tpms(
    node_tpms: Sequence[np.ndarray],
    current_state: Sequence[int],
    system_indices: Iterable[int],
) -> list[np.ndarray]:
    """Per-node backward (cause) TPMs for a subsystem in ``current_state``.

    Mixed-radix analogue of ``pyphi.tpm.backward_tpm``. With ``u`` the full
    current state, ``s`` the previous state of the system nodes and ``w`` the
    previous state of the background nodes, each returned array is

    .. math::

        \\sum_{w} P(X_i' = x_i \\mid s, w)
        \\frac{\\sum_{s'} P(u \\mid s', w)}{\\sum_{s', w'} P(u \\mid s', w')}

    so the background is weighted by how likely it is to have produced the
    current state, not averaged uniformly. Background axes come back as
    singletons. With no background this reduces to the forward TPM.

    Raises:
        StateUnreachableError: if no previous state leads to ``current_state``.
    """
    ks = tuple(node_tpms[0].shape[:-1])
    n = len(ks)
    system = tuple(sorted(int(i) for i in system_indices))
    background = tuple(j for j in range(n) if j not in system)
    current_state = tuple(int(s) for s in current_state)

    # P(u_t | s_{t-1}, w_{t-1}) over every previous state.
    pr_current = np.ones(ks, dtype=float)
    for i, tpm in enumerate(node_tpms):
        pr_current = pr_current * tpm[..., current_state[i]]
    normalization = float(pr_current.sum())
    if normalization == 0.0:
        raise StateUnreachableError(current_state)
    # Σ_{s_{t-1}} P(u_t | s_{t-1}, w_{t-1}); singleton on system axes.
    pr_given_background = pr_current.sum(axis=system, keepdims=True)
    weight = (pr_given_background / normalization)[..., np.newaxis]
    return [(tpm * weight).sum(axis=background, keepdims=True) for tpm in node_tpms]
