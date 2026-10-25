"""IIT-4.0 system Φ for MultivaluedSubsystem (mirrors ``new_big_phi.sia``)."""

from __future__ import annotations

from typing import Optional

from pyphi import conf, connectivity, utils
from pyphi.conf import config, fallback
from pyphi.models.subsystem import CauseEffectStructure
from pyphi.new_big_phi import (
    NullSystemIrreducibilityAnalysis,
    PhiStructure,
    ShortCircuitConditions,
    SystemIrreducibilityAnalysis,
    evaluate_partition,
    sia_minimization_key,
    system_intrinsic_information,
)
from pyphi.parallel import MapReduce
from pyphi.partition import system_partitions
from pyphi.relations import relations as compute_relations

from .subsystem_mv import MultivaluedSubsystem


def sia(
    subsystem: MultivaluedSubsystem,
    repertoire_distance: Optional[str] = None,
    directions=None,
    partition_scheme: Optional[str] = None,
    partitions=None,
    system_state=None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Minimum-information partition Φ for a multivalued subsystem.

    Same control flow as ``pyphi.new_big_phi.sia``; repertoires come from
    :class:`MultivaluedSubsystem` (mixed radix, no binary SBN).
    """
    partition_scheme = fallback(partition_scheme, config.SYSTEM_PARTITION_TYPE)

    def _null_sia(**kw):
        return NullSystemIrreducibilityAnalysis(
            system_state=system_state,
            node_indices=subsystem.node_indices,
            node_labels=subsystem.node_labels,
            **kw,
        )

    if not subsystem:
        return _null_sia(reasons=[ShortCircuitConditions.NO_SUBSYSTEM])

    if not connectivity.is_strong(subsystem.cm, subsystem.node_indices):
        return _null_sia(reasons=[ShortCircuitConditions.NO_STRONG_CONNECTIVITY])

    if len(subsystem.cut_indices) == 1:
        if not subsystem.cm[subsystem.node_indices][subsystem.node_indices]:
            return _null_sia(reasons=[ShortCircuitConditions.MONAD_WITH_NO_SELFLOOP])
        if not config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI:
            return _null_sia(
                reasons=[
                    ShortCircuitConditions.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI
                ]
            )

    if partitions is None:
        partitions = system_partitions(
            subsystem.node_indices,
            node_labels=subsystem.node_labels,
            partition_scheme=partition_scheme,
        )

    if system_state is None:
        system_state = system_intrinsic_information(
            subsystem, directions=directions
        )

    if config.SHORTCIRCUIT_SIA:
        from pyphi.new_big_phi import _has_no_cause_or_effect

        shortcircuit_reasons = _has_no_cause_or_effect(system_state)
        if shortcircuit_reasons:
            return _null_sia(reasons=shortcircuit_reasons)

    default_sia = _null_sia(reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS])

    parallel_kwargs = conf.parallel_kwargs(config.PARALLEL_CUT_EVALUATION, **kwargs)
    sias = MapReduce(
        evaluate_partition,
        partitions,
        map_kwargs=dict(
            subsystem=subsystem,
            system_state=system_state,
            repertoire_distance=repertoire_distance,
            directions=directions,
        ),
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating partitions",
        **parallel_kwargs,
    ).run()

    if not sias:
        return default_sia

    best = min(sias, key=sia_minimization_key)
    return best


def exact_phi(network, state) -> float:
    """Exact IIT-4.0 system Φ for a full-system multivalued state."""
    sub = MultivaluedSubsystem(network, state)
    return float(sia(sub).phi)


def phi_structure(network, state) -> PhiStructure:
    """IIT-4.0 Φ-structure (system Φ, distinctions, and relations).

    Distinctions are the irreducible cause-effect concepts of the full
    system. Relations are computed with the stock relation engine, which is
    radix-independent once purview units carry their states.
    """
    subsystem = MultivaluedSubsystem(network, state)
    analysis = sia(subsystem)
    distinctions = subsystem.all_distinctions()
    if not isinstance(distinctions, CauseEffectStructure):
        distinctions = CauseEffectStructure(distinctions)
    if analysis.system_state is not None:
        distinctions = distinctions.resolve_congruence(analysis.system_state)
    relation_structure = compute_relations(distinctions)
    return PhiStructure(
        sia=analysis,
        distinctions=distinctions,
        relations=relation_structure,
    )
