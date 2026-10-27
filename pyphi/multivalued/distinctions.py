"""Mechanism irreducibility (distinctions) for mixed-radix subsystems.

Mirrors the IIT 4.0 path of ``pyphi.subsystem.Subsystem`` (``find_mip``,
``find_mice``, ``distinction``) but calls the multivalued repertoire methods.
Partition search itself is combinatorial and does not assume radix 2.
"""

from __future__ import annotations

from pyphi import conf, resolve_ties, utils, validate
from pyphi.conf import config
from pyphi.direction import Direction
from pyphi.models import (
    CauseEffectStructure,
    Concept,
    MaximallyIrreducibleCause,
    MaximallyIrreducibleEffect,
    _null_ria,
)
from pyphi.models.mechanism import ShortCircuitConditions
from pyphi.network import irreducible_purviews
from pyphi.parallel import MapReduce
from pyphi.partition import mip_partitions


class DistinctionMethods:
    """MIP / MICE / distinction methods mixed into ``MultivaluedSubsystem``."""

    def _null_mip(self, direction, mechanism, purview, state=None, **kwargs):
        return _null_ria(
            direction,
            mechanism,
            purview,
            specified_state=state,
            **kwargs,
        )

    def _find_mip_single_state(
        self,
        specified_state,
        direction,
        mechanism,
        purview,
        repertoire,
        partitions,
        parallel_kwargs,
        **kwargs,
    ):
        partitions = (
            mip_partitions(mechanism, purview, self.node_labels)
            if partitions is None
            else partitions
        )

        def _evaluate_partition(partition):
            return self.evaluate_partition(
                direction,
                mechanism,
                purview,
                partition,
                repertoire=repertoire,
                state=specified_state,
                **kwargs,
            )

        candidate_mips = MapReduce(
            _evaluate_partition,
            partitions,
            shortcircuit_func=utils.is_falsy,
            desc="Evaluating mechanism partitions",
            **parallel_kwargs,
        ).run()

        ties = tuple(
            resolve_ties.partitions(
                candidate_mips,
                default=self._null_mip(
                    direction,
                    mechanism,
                    purview,
                    state=specified_state,
                    phi=0,
                ),
            )
        )
        for tie in ties:
            tie.set_partition_ties(ties)
        return ties[0]

    def find_mip(
        self, direction, mechanism, purview, partitions=None, state=None, **kwargs
    ):
        """Minimum-information partition of ``mechanism`` over ``purview``."""

        def null_mip(**null_kwargs):
            return self._null_mip(
                direction, mechanism, purview, state=state, **null_kwargs
            )

        if not purview:
            return null_mip(reasons=(ShortCircuitConditions.EMPTY_PURVIEW,))

        repertoire = self.repertoire(direction, mechanism, purview)
        if partitions is not None:
            partitions = list(partitions)

        parallel_kwargs = conf.parallel_kwargs(
            config.PARALLEL_MECHANISM_PARTITION_EVALUATION, **kwargs
        )
        if state is None:
            specified_states = self.intrinsic_information(
                direction, mechanism, purview
            ).ties
        else:
            specified_states = [state]

        mips = MapReduce(
            self._find_mip_single_state,
            specified_states,
            map_kwargs=dict(
                direction=direction,
                mechanism=mechanism,
                purview=purview,
                repertoire=repertoire,
                partitions=partitions,
                parallel_kwargs=parallel_kwargs,
            ),
            desc="Finding MIP for maximum intrinsic information states",
            **parallel_kwargs,
        ).run()

        ties = tuple(resolve_ties.states(mips))
        for tie in ties:
            tie.set_state_ties(ties)
        return ties[0]

    def potential_purviews(self, direction, mechanism, purviews=None):
        """Purviews that are not trivially reducible for ``mechanism``."""
        _potential = set(self.network.potential_purviews(direction, mechanism))
        if purviews is None:
            purviews = _potential
        else:
            purviews = _potential & set(purviews)
        purviews = [
            purview
            for purview in purviews
            if set(purview).issubset(self.node_indices)
        ]
        return irreducible_purviews(self.cm, direction, mechanism, purviews)

    def find_mice(self, direction, mechanism, purviews=None, **kwargs):
        """Maximally irreducible cause or effect of ``mechanism``."""
        purviews = self.potential_purviews(direction, mechanism, purviews)
        if direction == Direction.CAUSE:
            mice_class = MaximallyIrreducibleCause
        elif direction == Direction.EFFECT:
            mice_class = MaximallyIrreducibleEffect
        else:
            return validate.direction(direction)

        no_purviews = mice_class(
            self._null_mip(
                direction,
                mechanism,
                (),
                reasons=(ShortCircuitConditions.NO_PURVIEWS,),
            )
        )
        if not purviews:
            return no_purviews

        def _find_mip(purview):
            return self.find_mip(direction, mechanism, purview)

        parallel_kwargs = conf.parallel_kwargs(
            config.PARALLEL_PURVIEW_EVALUATION, **kwargs
        )
        map_reduce = MapReduce(
            _find_mip,
            purviews,
            total=len(purviews),
            desc="Evaluating purviews",
            **parallel_kwargs,
        )
        all_mice = map(mice_class, map_reduce.run())
        ties = tuple(resolve_ties.purviews(all_mice, default=no_purviews))
        for tie in ties:
            tie.set_purview_ties(ties)
        return ties[0]

    def distinction(self, mechanism):
        """The concept specified by ``mechanism`` (cause and effect MICE)."""
        if not mechanism:
            raise ValueError("distinction requires a non-empty mechanism")
        cause = self.find_mice(Direction.CAUSE, mechanism)
        effect = self.find_mice(Direction.EFFECT, mechanism)
        return Concept(mechanism=mechanism, cause=cause, effect=effect)

    def all_distinctions(self, **kwargs):
        """Every positive-φ distinction in this subsystem."""
        mechanisms = utils.powerset(self.node_indices, nonempty=True)
        distinctions = filter(
            None,
            (self.distinction(mechanism, **kwargs) for mechanism in mechanisms),
        )
        return CauseEffectStructure(distinctions)
