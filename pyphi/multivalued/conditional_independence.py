"""Mixed-radix conditional independence on state-by-state TPMs.

Never routes through binary state-by-node conversion or ``int(log2(...))``.
A next-state row is CI iff it equals the product of its per-node
categorical marginals (little-endian mixed radix).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def decode_state(index: int, num_states_per_node: Sequence[int]) -> tuple[int, ...]:
    """Little-endian mixed-radix decode (node 0 = least significant)."""
    state = []
    rem = int(index)
    for k in num_states_per_node:
        state.append(rem % k)
        rem //= k
    if rem != 0:
        raise ValueError(
            f"index {index} out of range for num_states_per_node={num_states_per_node}"
        )
    return tuple(state)


def encode_state(state: Sequence[int], num_states_per_node: Sequence[int]) -> int:
    """Little-endian mixed-radix encode."""
    if len(state) != len(num_states_per_node):
        raise ValueError("state length must match num_states_per_node")
    idx = 0
    place = 1
    for s, k in zip(state, num_states_per_node):
        if not (0 <= int(s) < int(k)):
            raise ValueError(f"state value {s} out of range for alphabet {k}")
        idx += int(s) * place
        place *= int(k)
    return idx


def node_marginals(
    row: np.ndarray, num_states_per_node: Sequence[int]
) -> list[np.ndarray]:
    """Per-node next-state marginals for one SBS row."""
    n_states = int(np.prod(num_states_per_node))
    if row.shape != (n_states,):
        raise ValueError(f"row shape {row.shape} != ({n_states},)")
    marginals = []
    for node, k in enumerate(num_states_per_node):
        m = np.zeros(k, dtype=float)
        for j in range(n_states):
            st = decode_state(j, num_states_per_node)
            m[st[node]] += row[j]
        marginals.append(m)
    return marginals


def reconstruct_joint(marginals: Sequence[np.ndarray]) -> np.ndarray:
    """Product of categorical marginals over the mixed-radix joint."""
    ks = [len(m) for m in marginals]
    n_states = int(np.prod(ks))
    joint = np.ones(n_states, dtype=float)
    for j in range(n_states):
        st = decode_state(j, ks)
        for node, s in enumerate(st):
            joint[j] *= marginals[node][s]
    return joint


def row_conditionally_independent(
    row: np.ndarray, num_states_per_node: Sequence[int], atol: float = 1e-12
) -> bool:
    marg = node_marginals(row, num_states_per_node)
    return bool(np.allclose(row, reconstruct_joint(marg), atol=atol))


def sbs_conditionally_independent(
    sbs: np.ndarray, num_states_per_node: Sequence[int], atol: float = 1e-12
) -> bool:
    """Return True iff every row factors into per-node categorical marginals."""
    sbs = np.asarray(sbs, dtype=float)
    n_states = int(np.prod(num_states_per_node))
    if sbs.shape != (n_states, n_states):
        raise ValueError(
            f"SBS shape {sbs.shape} incompatible with "
            f"num_states_per_node={tuple(num_states_per_node)} "
            f"(expected {(n_states, n_states)})"
        )
    return all(
        row_conditionally_independent(sbs[i], num_states_per_node, atol=atol)
        for i in range(n_states)
    )
