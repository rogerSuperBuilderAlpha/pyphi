# -*- coding: utf-8 -*-
"""Multivalued Subsystem — IIT-4.0 repertoires over mixed-radix SBS TPMs."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from pyphi import connectivity, metrics
from pyphi import utils as pyphi_utils
from pyphi import validate
from pyphi.conf import config, fallback
from pyphi.connectivity import get_inputs_from_cm
from pyphi.direction import Direction
from pyphi.models.cuts import NullCut
from pyphi.models.mechanism import RepertoireIrreducibilityAnalysis, StateSpecification

from .conditional_independence import encode_state
from .distinctions import DistinctionMethods
from .network import MultivaluedNetwork
from .tpm import MultivaluedTPMError
from .utils_mv import (
    all_states_mixed,
    cut_node_tpm,
    factorize_node_tpms,
    max_entropy_mv,
    repertoire_shape_mv,
    state_of,
)


class MultivaluedSubsystem(DistinctionMethods):
    """Candidate system for exact IIT-4.0 Φ on a :class:`MultivaluedNetwork`.

    Duck-types the surface ``new_big_phi.sia`` needs, but computes repertoires
    from SBS / categorical node TPMs — never binary SBN or ``int(log2)``.
    """

    def __init__(
        self,
        network: MultivaluedNetwork,
        state: Sequence[int],
        nodes=None,
        cut=None,
    ):
        if not isinstance(network, MultivaluedNetwork):
            raise TypeError('MultivaluedSubsystem requires MultivaluedNetwork')
        self.network = network
        self.node_labels = network.node_labels
        self.node_indices = self.node_labels.coerce_to_indices(nodes)
        self.num_states_per_node = network.num_states_per_node
        self.state = tuple(int(s) for s in state)

        if len(self.state) != network.size:
            raise ValueError(
                f"state length {len(self.state)} != network size {network.size}"
            )
        for i, (s, k) in enumerate(zip(self.state, self.num_states_per_node)):
            if not (0 <= s < k):
                raise ValueError(
                    f"state[{i}]={s} out of range for alphabet size {k}"
                )

        self.cut = (
            cut
            if cut is not None
            else NullCut(self.node_indices, self.node_labels)
        )
        self.cm = self.cut.apply_cut(network.cm)
        self.proper_cm = connectivity.subadjacency(self.cm, self.node_indices)

        # Background nodes fixed to their current state (stock Subsystem pattern).
        self.external_indices = tuple(
            sorted(set(network.node_indices) - set(self.node_indices))
        )
        self._background = {
            i: self.state[i] for i in self.external_indices
        }

        sbs = network.sbs()
        raw_node_tpms = factorize_node_tpms(sbs, self.num_states_per_node)
        self._effect_by_node = {}
        for i in self.node_indices:
            inputs = set(get_inputs_from_cm(i, self.cm))
            self._effect_by_node[i] = cut_node_tpm(
                raw_node_tpms[i], i, inputs, self.num_states_per_node
            )

        # Reachability: some previous state must map to current with p>0.
        curr_idx = encode_state(self.state, self.num_states_per_node)
        col = sbs[:, curr_idx].astype(float)
        if col.sum() <= 0.0:
            from pyphi.exceptions import StateUnreachableError

            raise StateUnreachableError(self.state)

        self.nodes = tuple(self.node_indices)

    def _merge_background(self, condition: dict[int, int]) -> dict[int, int]:
        merged = dict(self._background)
        merged.update(condition)
        return merged

    @property
    def proper_state(self) -> tuple[int, ...]:
        return state_of(self.node_indices, self.state)

    @property
    def cut_indices(self):
        return self.node_indices

    def __bool__(self):
        return bool(self.node_indices)

    def __len__(self):
        return len(self.node_indices)

    def apply_cut(self, cut):
        return MultivaluedSubsystem(
            self.network, self.state, nodes=self.node_indices, cut=cut
        )

    # --- repertoires ---

    def _effect_single_node(
        self, condition: dict[int, int], purview_node: int
    ) -> np.ndarray:
        tpm = self._effect_by_node[purview_node]
        ks = self.num_states_per_node
        k_p = ks[purview_node]
        condition = self._merge_background(condition)
        free = [j for j in range(len(ks)) if j not in condition]
        acc = np.zeros(k_p, dtype=float)
        free_states = list(all_states_mixed([ks[j] for j in free])) if free else [()]
        n_free = len(free_states)
        for free_st in free_states:
            prev = [0] * len(ks)
            for j, s in condition.items():
                prev[j] = s
            for j, s in zip(free, free_st):
                prev[j] = s
            acc += tpm[tuple(prev)]
        acc /= float(n_free)
        out = np.zeros(
            repertoire_shape_mv(
                self.network.node_indices, (purview_node,), ks
            ),
            dtype=float,
        )
        for s in range(k_p):
            idx = [0] * len(ks)
            idx[purview_node] = s
            out[tuple(idx)] = acc[s]
        return out

    def effect_repertoire(
        self,
        mechanism,
        purview,
        mechanism_state=None,
        direction=Direction.EFFECT,
    ):
        if not purview:
            return np.array([1.0])
        if mechanism_state is None:
            mechanism_state = state_of(mechanism, self.state)
        condition = dict(zip(mechanism, mechanism_state))
        joint = np.ones(
            repertoire_shape_mv(
                self.network.node_indices, purview, self.num_states_per_node
            ),
            dtype=float,
        )
        for p in purview:
            joint = joint * self._effect_single_node(condition, p)
        return joint

    def cause_repertoire(self, mechanism, purview, **kwargs):
        """Cause repertoire as the normalized product of single-node causes.

        Matches stock IIT 4.0 on a full system: each mechanism node contributes
        ``P(X_m = s_m | previous)``, inputs outside the purview are averaged
        out, and the product is normalized. Per-node alphabets may differ.
        Non-input axes are already uniform in ``_effect_by_node``, so averaging
        them does not change the conditional.
        """
        if not purview:
            return np.array([1.0])
        ks = self.num_states_per_node
        shape = repertoire_shape_mv(self.network.node_indices, purview, ks)
        if not mechanism:
            return max_entropy_mv(self.network.node_indices, purview, ks)
        purview_set = set(purview)
        joint = np.ones(shape, dtype=float)
        n = len(ks)
        non_purview = tuple(j for j in range(n) if j not in purview_set)
        for m in mechanism:
            # P(X_m' = current state | previous). Shape ``ks``.
            p = np.asarray(self._effect_by_node[m][..., self.state[m]], dtype=float)
            if non_purview:
                p = p.mean(axis=non_purview, keepdims=True)
            joint = joint * p
        total = float(joint.sum())
        if total == 0.0:
            return joint
        return joint / total

    def repertoire(self, direction, mechanism, purview, **kwargs):
        if direction == Direction.CAUSE:
            return self.cause_repertoire(mechanism, purview, **kwargs)
        if direction == Direction.EFFECT:
            return self.effect_repertoire(mechanism, purview, **kwargs)
        return validate.direction(direction)

    def unconstrained_repertoire(self, direction, purview, **kwargs):
        return self.repertoire(direction, (), purview, **kwargs)

    def forward_effect_repertoire(self, mechanism, purview, **kwargs):
        return self.effect_repertoire(mechanism, purview, **kwargs)

    def forward_effect_probability(
        self, mechanism, purview, purview_state, **kwargs
    ):
        rep = self.forward_effect_repertoire(mechanism, purview, **kwargs)
        return float(rep.squeeze()[purview_state] if purview else 1.0)

    def forward_cause_repertoire(
        self, mechanism, purview, purview_state=None
    ):
        # Mirror stock repertoire.forward_cause_repertoire with mixed radix
        mechanism_state = state_of(mechanism, self.state)
        if purview:
            shape = [self.num_states_per_node[i] for i in purview]
            repertoire = np.empty(shape, dtype=float)
            if purview_state is None:
                purview_states = list(
                    all_states_mixed(
                        [self.num_states_per_node[i] for i in purview]
                    )
                )
            else:
                purview_states = [purview_state]
                repertoire = np.zeros(shape, dtype=float)
        else:
            repertoire = np.array([1.0])
            purview_states = [()]
        for ps in purview_states:
            # forward cause: effect repertoire with roles switched
            er = self.effect_repertoire(
                mechanism=purview,
                purview=mechanism,
                mechanism_state=ps,
                direction=Direction.CAUSE,
            )
            val = float(er.squeeze()[mechanism_state]) if mechanism else 1.0
            if purview:
                repertoire[ps] = val
            else:
                repertoire = np.array([val])
        if purview:
            full = np.zeros(
                repertoire_shape_mv(
                    self.network.node_indices,
                    purview,
                    self.num_states_per_node,
                ),
                dtype=float,
            )
            # embed squeezed repertoire into network-shaped array
            for ps in all_states_mixed(
                [self.num_states_per_node[i] for i in purview]
            ):
                idx = [0] * len(self.num_states_per_node)
                for j, s in zip(purview, ps):
                    idx[j] = s
                full[tuple(idx)] = repertoire[ps]
            return full
        return repertoire

    def forward_cause_probability(
        self, mechanism, purview, purview_state, **kwargs
    ):
        er = self.effect_repertoire(
            mechanism=purview,
            purview=mechanism,
            mechanism_state=purview_state,
        )
        mechanism_state = state_of(mechanism, self.state)
        return float(er.squeeze()[mechanism_state]) if mechanism else 1.0

    def forward_probability(
        self, direction, mechanism, purview, purview_state, **kwargs
    ):
        if direction == Direction.CAUSE:
            return self.forward_cause_probability(
                mechanism, purview, purview_state, **kwargs
            )
        if direction == Direction.EFFECT:
            return self.forward_effect_probability(
                mechanism, purview, purview_state, **kwargs
            )
        return validate.direction(direction)

    def forward_repertoire(
        self, direction, mechanism, purview, purview_state, **kwargs
    ):
        if direction == Direction.CAUSE:
            return self.forward_cause_repertoire(
                mechanism, purview, purview_state
            )
        if direction == Direction.EFFECT:
            return self.forward_effect_repertoire(mechanism, purview, **kwargs)
        return validate.direction(direction)

    def unconstrained_forward_effect_repertoire(self, mechanism, purview):
        ks_m = [self.num_states_per_node[i] for i in mechanism]
        reps = [
            self.forward_effect_repertoire(
                mechanism, purview, mechanism_state=st
            )
            for st in all_states_mixed(ks_m)
        ] if mechanism else [self.forward_effect_repertoire(mechanism, purview)]
        return np.mean(np.stack(reps), axis=0)

    def unconstrained_forward_cause_repertoire(self, mechanism, purview):
        mean_p = self.forward_cause_repertoire(mechanism, purview, None).mean()
        out = np.empty(
            repertoire_shape_mv(
                self.network.node_indices, purview, self.num_states_per_node
            )
        )
        out.fill(mean_p)
        return out

    def unconstrained_forward_repertoire(self, direction, mechanism, purview):
        if direction == Direction.CAUSE:
            return self.unconstrained_forward_cause_repertoire(
                mechanism, purview
            )
        if direction == Direction.EFFECT:
            return self.unconstrained_forward_effect_repertoire(
                mechanism, purview
            )
        return validate.direction(direction)

    def partitioned_repertoire(
        self, direction, partition, repertoire_distance=None, **kwargs
    ):
        repertoire_distance = fallback(
            repertoire_distance, config.REPERTOIRE_DISTANCE
        )
        if repertoire_distance == 'GENERALIZED_INTRINSIC_DIFFERENCE':
            if 'state' not in kwargs:
                raise ValueError(
                    'must provide purview state for generalized intrinsic difference'
                )
            purview_state = kwargs.pop('state')
            prs = [
                self.forward_probability(
                    direction,
                    part.mechanism,
                    part.purview,
                    purview_state=pyphi_utils.substate(
                        partition.purview, purview_state, part.purview
                    ),
                    **kwargs,
                )
                for part in partition
            ]
            return float(np.prod(prs))
        repertoires = [
            self.repertoire(direction, part.mechanism, part.purview, **kwargs)
            for part in partition
        ]
        return np.multiply.reduce(repertoires)

    def intrinsic_information(
        self,
        direction: Direction,
        mechanism: Tuple[int],
        purview: Tuple[int],
        repertoire_distance: str = None,
        states: Iterable[Iterable[int]] = None,
    ):
        repertoire_distance = fallback(
            repertoire_distance, config.REPERTOIRE_DISTANCE_INFORMATION
        )
        if states is None:
            states = list(
                all_states_mixed(
                    [self.num_states_per_node[i] for i in purview]
                )
            ) if purview else [()]

        if repertoire_distance == 'GENERALIZED_INTRINSIC_DIFFERENCE':
            selectivity_repertoire = self.repertoire(
                direction, mechanism, purview
            )
            repertoire = self.forward_repertoire(
                direction, mechanism, purview, None
            )
            unconstrained_repertoire = self.unconstrained_forward_repertoire(
                direction, mechanism, purview
            )
            gid = metrics.distribution.generalized_intrinsic_difference(
                repertoire,
                unconstrained_repertoire,
                selectivity_repertoire,
            ).squeeze()

            def evaluate_state(state):
                return gid[state] if purview else float(gid)

        else:
            repertoire = self.repertoire(direction, mechanism, purview)
            unconstrained_repertoire = self.unconstrained_repertoire(
                direction, purview
            )

            def evaluate_state(state):
                return metrics.distribution.repertoire_distance(
                    repertoire, unconstrained_repertoire, state=state
                )

        state_to_information = {state: evaluate_state(state) for state in states}
        max_information = max(state_to_information.values())
        ties = [
            StateSpecification(
                direction=direction,
                purview=purview,
                state=state,
                intrinsic_information=information,
                repertoire=repertoire,
                unconstrained_repertoire=unconstrained_repertoire,
            )
            for state, information in state_to_information.items()
            if information == max_information
        ]
        for tie in ties:
            tie.set_ties(ties)
        return ties[0]

    def evaluate_partition(
        self,
        direction,
        mechanism,
        purview,
        partition,
        repertoire=None,
        partitioned_repertoire=None,
        repertoire_distance=None,
        partitioned_repertoire_kwargs=None,
        **kwargs,
    ):
        repertoire_distance = fallback(
            repertoire_distance, config.REPERTOIRE_DISTANCE
        )
        if repertoire is None:
            repertoire = self.repertoire(direction, mechanism, purview)
        if repertoire_distance == 'GENERALIZED_INTRINSIC_DIFFERENCE':
            purview_state = kwargs['state'].state
            selectivity = repertoire.squeeze()[purview_state] if purview else float(
                repertoire.squeeze()
            )
            forward_pr = self.forward_probability(
                direction, mechanism, purview, purview_state
            )
            if partitioned_repertoire is None:
                partitioned_pr = self.partitioned_repertoire(
                    direction, partition, state=purview_state
                )
            else:
                partitioned_pr = partitioned_repertoire
            phi = metrics.distribution.generalized_intrinsic_difference(
                forward_repertoire=forward_pr,
                partitioned_forward_repertoire=partitioned_pr,
                selectivity_repertoire=selectivity,
            )
            return RepertoireIrreducibilityAnalysis(
                phi=phi,
                direction=direction,
                mechanism=mechanism,
                purview=purview,
                partition=partition,
                repertoire=forward_pr,
                partitioned_repertoire=partitioned_pr,
                mechanism_state=state_of(mechanism, self.state),
                purview_state=state_of(purview, self.state),
                specified_state=kwargs.get('state'),
                node_labels=self.node_labels,
                selectivity=selectivity,
            )
        raise MultivaluedTPMError(
            f"M2 MultivaluedSubsystem only implements "
            f"GENERALIZED_INTRINSIC_DIFFERENCE; got {repertoire_distance}"
        )
