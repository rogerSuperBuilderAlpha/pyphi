# -*- coding: utf-8 -*-
"""Multivalued Network for the opt-in IIT 4.0 mixed-radix path."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pyphi import cache, connectivity
from pyphi.labels import NodeLabels
from pyphi.network import Network

from .tpm import MultivaluedExplicitTPM


class MultivaluedNetwork(Network):
    """Network that accepts mixed-radix SBS TPMs via ``num_states_per_node``.

    Subclasses stock ``pyphi.Network`` so ``isinstance(..., Network)`` holds,
    but stores a :class:`MultivaluedExplicitTPM` and never converts through
    binary state-by-node form.

    State spaces stay the product of per-node radices. Nothing on this path
    calls ``int(log2(S))`` or ``state_by_state2state_by_node``.
    """

    def __init__(
        self,
        tpm,
        num_states_per_node: Sequence[int],
        cm=None,
        node_labels=None,
        purview_cache=None,
        validate: bool = True,
        check_independence: bool = True,
    ):
        if isinstance(tpm, MultivaluedExplicitTPM):
            if tuple(tpm.num_states_per_node) != tuple(num_states_per_node):
                raise ValueError(
                    'tpm.num_states_per_node does not match constructor argument'
                )
            self._tpm = tpm
        else:
            self._tpm = MultivaluedExplicitTPM(
                tpm,
                num_states_per_node,
                validate=validate,
                check_independence=check_independence,
            )

        self._num_states_per_node = self._tpm.num_states_per_node
        self._cm, self._cm_hash = self._build_cm(cm)
        self._node_indices = tuple(range(self.size))
        self._node_labels = NodeLabels(node_labels, self._node_indices)
        self.purview_cache = purview_cache or cache.PurviewCache()

        # Stock validate.network re-calls ExplicitTPM.validate and assumes
        # binary SBN layout; validate CM shape ourselves only.
        if self._cm.shape[0] != self.size:
            raise ValueError(
                'Connectivity matrix must be NxN, where N is the '
                'number of nodes in the network.'
            )

    @property
    def num_states_per_node(self) -> tuple[int, ...]:
        return self._num_states_per_node

    @property
    def num_states(self) -> int:
        """Product of alphabet sizes (mixed radix), not ``2**size``."""
        return self._tpm.num_states

    @property
    def size(self) -> int:
        return len(self._num_states_per_node)

    def __len__(self) -> int:
        return self.size

    @property
    def causally_significant_nodes(self):
        return connectivity.causally_significant_nodes(self.cm)

    def sbs(self) -> np.ndarray:
        """Preserved state-by-state TPM (copy)."""
        return self._tpm.sbs_copy()

    def __repr__(self) -> str:
        return (
            f"MultivaluedNetwork(size={self.size}, "
            f"num_states_per_node={self._num_states_per_node}, "
            f"tpm.shape={self.tpm.shape})"
        )
