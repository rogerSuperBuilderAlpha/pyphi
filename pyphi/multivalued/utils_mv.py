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

    Each node's marginal is divided by the row sum. The row sum is 1 up to
    rounding, so this changes values by at most an ulp or so, but it restores
    exact 0 and 1 where the state-by-state entries were products of exact 0
    and 1 with inexact factors (a deterministic node next to noisy ones).
    Those exact values decide reachability and short-circuits, so they must
    survive the state-by-node -> state-by-state -> marginal round trip.
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
            row_sum = tpm_i[prev].sum()
            if row_sum != 0.0 and row_sum != 1.0:
                tpm_i[prev] /= row_sum
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


def node_tpm_columns(node_tpm: np.ndarray) -> np.ndarray:
    """Columns of a node TPM in the form stock carries them.

    Stock pyphi carries a binary node as a single P(on) column (state-by-node
    form) and derives P(off) = 1 - P(on) in ``Node``; a node with k > 2 states
    keeps all k columns. Keeping to that convention makes the radix-2 path's
    floating-point operations identical to stock's.
    """
    if node_tpm.shape[-1] == 2:
        return node_tpm[..., 1:2]
    return node_tpm


def probability_of_state(columns: np.ndarray, state: int) -> np.ndarray:
    """P(X_i' = state | previous) from a node's columns (see above)."""
    if columns.shape[-1] == 1:
        on = columns[..., 0]
        return on if state else 1 - on
    return columns[..., state]


def stack_node_tpm(
    columns: np.ndarray, non_inputs: Iterable[int], k: int
) -> np.ndarray:
    """Finish a node TPM the way ``pyphi.node.Node`` does.

    Marginalize the non-input axes out of each column (``marginalize_out``:
    sum, then divide by the product of the axis sizes), and for a binary node
    stack ``[1 - on, on]`` on the last axis.
    """
    if columns.shape[-1] == 1:
        on = marginalize_out_mv(columns[..., 0], non_inputs)
        return np.stack([1 - on, on], axis=-1)
    assert columns.shape[-1] == k
    return marginalize_out_mv(columns, non_inputs)


def backward_node_tpms(
    node_columns: Sequence[np.ndarray],
    current_state: Sequence[int],
    system_indices: Iterable[int],
) -> list[np.ndarray]:
    """Per-node backward (cause) TPM columns for a subsystem in ``current_state``.

    Mixed-radix analogue of ``pyphi.tpm.backward_tpm``, operation for
    operation. With ``u`` the full current state, ``s`` the previous state of
    the system nodes and ``w`` the previous state of the background nodes,
    each returned array is

    .. math::

        \\sum_{w} P(X_i' = x_i \\mid s, w)
        \\frac{\\sum_{s'} P(u \\mid s', w)}{\\sum_{s', w'} P(u \\mid s', w')}

    so the background is weighted by how likely it is to have produced the
    current state, not averaged uniformly. Background axes come back as
    singletons. With no background this reduces to the forward TPM.

    Raises:
        StateUnreachableError: if no previous state leads to ``current_state``.
    """
    n = len(node_columns)
    system = tuple(sorted(int(i) for i in system_indices))
    background = tuple(j for j in range(n) if j not in system)
    current_state = tuple(int(s) for s in current_state)

    # p(u_t | s_{t-1}, w_{t-1}); stock's ``probability_of_current_state``.
    state_probabilities = np.stack(
        [
            probability_of_state(col, current_state[i])
            for i, col in enumerate(node_columns)
        ],
        axis=-1,
    )
    pr_current_state = state_probabilities.prod(axis=-1, keepdims=True)
    # Σ_{s_{t-1}} p(u_t | s_{t-1}, w_{t-1})
    pr_current_state_given_only_background = pr_current_state.sum(
        axis=system, keepdims=True
    )
    normalization = np.sum(pr_current_state)
    if normalization == 0.0:
        raise StateUnreachableError(current_state)
    return [
        (col * pr_current_state_given_only_background / normalization).sum(
            axis=background, keepdims=True
        )
        for col in node_columns
    ]
