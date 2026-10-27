"""Opt-in multivalued IIT 4.0: state space, binary parity, and Φ-structure."""

import math

import numpy as np
import pytest

from pyphi import Network, Subsystem, convert, new_big_phi
from pyphi.conf import config
from pyphi.exceptions import ConditionallyDependentError
from pyphi.multivalued import (
    MultivaluedExplicitTPM,
    MultivaluedNetwork,
    exact_phi,
    maximal_complex,
    phi_structure,
)
from pyphi.multivalued.conditional_independence import encode_state
from pyphi.multivalued.utils_mv import all_states_mixed
from pyphi.tpm import ExplicitTPM

PHI_TOL = 1e-8

# W' = S, S' = W AND C, C' = S. Edges point from a node to the nodes it drives.
TRIAD_CM = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
)


@pytest.fixture(autouse=True)
def _quiet_pyphi():
    with config.override(PROGRESS_BARS=False, PARALLEL=False):
        yield


def deterministic_sbs(num_states_per_node, next_fn):
    """Little-endian state-by-state TPM for a deterministic mixed-radix rule."""
    ks = tuple(int(k) for k in num_states_per_node)
    n_states = int(np.prod(ks))
    sbs = np.zeros((n_states, n_states), dtype=float)
    for state in all_states_mixed(ks):
        src = encode_state(state, ks)
        dst = encode_state(tuple(next_fn(state)), ks)
        sbs[src, dst] = 1.0
    return sbs


def binary_sbn(rules):
    """State-by-node TPM, rows little-endian, columns P(node = 1)."""
    n = len(rules)
    tpm = np.zeros((2**n, n), dtype=float)
    for index in range(2**n):
        state = tuple((index >> node) & 1 for node in range(n))
        for node, rule in enumerate(rules):
            tpm[index, node] = rule(state)
    return tpm


def binary_triad_rules():
    return (
        lambda state: state[1],
        lambda state: state[0] & state[2],
        lambda state: state[1],
    )


def ternary_triad_next(state):
    """Ternary analogue of the conjunctive triad: S' = min(W, C)."""
    worker, mediator, counterpart = state
    return (mediator, min(worker, counterpart), mediator)


def mixed_radix_next(state):
    """Binary–ternary–binary mediator. Middle node is the sum of the ends."""
    left, middle, right = state
    return (1 if middle > 0 else 0, left + right, 1 if middle > 0 else 0)


def stock_phi(sbn, cm, state, labels):
    network = Network(sbn, cm=cm, node_labels=labels)
    subsystem = Subsystem(network, state)
    return new_big_phi.sia(subsystem)


def mv_network_from_sbn(sbn, cm, labels):
    sbs = convert.state_by_node2state_by_state(sbn)
    radix = (2,) * sbn.shape[-1]
    return MultivaluedNetwork(sbs, radix, cm=cm, node_labels=labels), sbs


class TestStateSpacePreservation:
    """M1: SBS ingest must not collapse alphabets through ``int(log2(S))``."""

    def test_ternary_sbs_preserved_and_stock_collapses(self):
        sbs = deterministic_sbs((3, 3), lambda state: (state[1], state[0]))
        with pytest.raises(Exception):
            Network(sbs, node_labels=("A", "B"))

        collapsed = ExplicitTPM(sbs, validate=False)
        assert collapsed.is_state_by_state()
        # Stock reads the unit count off the binary log of the state count.
        assert collapsed.number_of_units == int(math.log2(sbs.shape[1]))
        assert collapsed.number_of_units == 3
        assert collapsed.number_of_units != 2

        native = MultivaluedExplicitTPM(sbs, (3, 3))
        network = MultivaluedNetwork(native, (3, 3), node_labels=("A", "B"))
        assert native.num_states_per_node == (3, 3)
        assert native.num_states == 9
        assert native.shape == (9, 9)
        assert network.size == 2
        assert network.num_states == 9
        assert network.tpm.shape == (9, 9)
        assert np.allclose(network.sbs(), sbs)

        triad = deterministic_sbs((3, 3, 3), ternary_triad_next)
        triad_net = MultivaluedNetwork(triad, (3, 3, 3), node_labels=("W", "S", "C"))
        assert triad_net.tpm.shape == (27, 27)
        assert triad_net.num_states == 27
        assert np.allclose(triad_net.sbs(), triad)

    def test_mixed_radix_sbs_preserved(self):
        radix = (2, 3, 4)
        sbs = deterministic_sbs(radix, lambda state: (state[0], state[1], state[2]))
        assert sbs.shape == (24, 24)
        network = MultivaluedNetwork(
            sbs, radix, cm=np.eye(3), node_labels=("A", "B", "C")
        )
        assert network.num_states_per_node == radix
        assert network.num_states == 24
        assert network.size == 3
        assert np.allclose(network.sbs(), sbs)
        # int(log2(24)) == 4 and 2**4 == 16, which is not this state space.
        assert int(math.log2(sbs.shape[0])) == 4
        assert 2 ** int(math.log2(sbs.shape[0])) != sbs.shape[0]

    def test_validation_and_conditional_independence(self):
        with pytest.raises(ValueError):
            MultivaluedExplicitTPM(np.eye(2), (1, 2))
        with pytest.raises(ValueError):
            MultivaluedExplicitTPM(np.eye(4), (2, 3))

        sbs = np.zeros((4, 4))
        row = np.zeros(4)
        row[encode_state((0, 0), (2, 2))] = 0.5
        row[encode_state((1, 1), (2, 2))] = 0.5
        sbs[:] = row
        with pytest.raises(ConditionallyDependentError):
            MultivaluedExplicitTPM(sbs, (2, 2))


class TestBinaryParity:
    """M2: every binary network must match stock ``new_big_phi``."""

    def test_triad_system_phi_is_two(self):
        rules = binary_triad_rules()
        sbn = binary_sbn(rules)
        state = (1, 1, 1)
        labels = ("W", "S", "C")
        stock = stock_phi(sbn, TRIAD_CM, state, labels)
        network, _sbs = mv_network_from_sbn(sbn, TRIAD_CM, labels)
        ours = exact_phi(network, state)
        assert float(stock.phi) == pytest.approx(2.0, abs=PHI_TOL)
        assert ours == pytest.approx(float(stock.phi), abs=PHI_TOL)
        assert ours == pytest.approx(2.0, abs=PHI_TOL)

    def test_reducible_controls_are_zero(self):
        # No edges: the full system is not strongly connected.
        sbn = binary_sbn(
            (lambda state: state[0], lambda state: state[1], lambda state: 0)
        )
        cm = np.zeros((3, 3))
        state = (1, 0, 0)
        labels = ("A", "B", "C")
        stock = stock_phi(sbn, cm, state, labels)
        network, _sbs = mv_network_from_sbn(sbn, cm, labels)
        assert float(stock.phi) == pytest.approx(0.0, abs=PHI_TOL)
        assert exact_phi(network, state) == pytest.approx(0.0, abs=PHI_TOL)

        # Two independent copy nodes, binary and then the same rule at k=2
        # inside the multivalued types.
        sbs = deterministic_sbs((2, 2), lambda state: state)
        indep = MultivaluedNetwork(sbs, (2, 2), cm=np.eye(2), node_labels=("A", "B"))
        assert exact_phi(indep, (1, 0)) == pytest.approx(0.0, abs=PHI_TOL)

    def test_phi_structure_matches_stock(self):
        rules = binary_triad_rules()
        sbn = binary_sbn(rules)
        state = (1, 1, 1)
        labels = ("W", "S", "C")
        stock_network = Network(sbn, cm=TRIAD_CM, node_labels=labels)
        stock_structure = new_big_phi.phi_structure(Subsystem(stock_network, state))
        network, _sbs = mv_network_from_sbn(sbn, TRIAD_CM, labels)
        ours = phi_structure(network, state)

        assert float(ours.phi) == pytest.approx(float(stock_structure.phi), abs=PHI_TOL)
        assert len(ours.distinctions) == len(stock_structure.distinctions)
        assert len(ours.distinctions) > 0
        assert ours.distinctions.sum_phi() == pytest.approx(
            stock_structure.distinctions.sum_phi(), abs=PHI_TOL
        )
        assert len(ours.relations) == len(stock_structure.relations)
        assert len(ours.relations) > 0
        assert ours.relations.sum_phi() == pytest.approx(
            stock_structure.relations.sum_phi(), abs=PHI_TOL
        )
        stock_mechanisms = sorted(
            concept.mechanism for concept in stock_structure.distinctions
        )
        our_mechanisms = sorted(concept.mechanism for concept in ours.distinctions)
        assert our_mechanisms == stock_mechanisms

        stock_complex = new_big_phi.maximal_complex(stock_network, state)
        our_complex = maximal_complex(network, state)
        assert float(our_complex.phi) == pytest.approx(
            float(stock_complex.phi), abs=PHI_TOL
        )
        assert tuple(our_complex.node_indices) == tuple(stock_complex.node_indices)
        assert tuple(our_complex.node_indices) == (0, 1, 2)


class TestTernaryAndMixedRadix:
    """Exact Φ_s, distinctions, relations, and complexes off radix 2."""

    def test_ternary_triad_phi_structure(self):
        sbs = deterministic_sbs((3, 3, 3), ternary_triad_next)
        assert sbs.shape == (27, 27)
        network = MultivaluedNetwork(
            sbs, (3, 3, 3), cm=TRIAD_CM, node_labels=("W", "S", "C")
        )
        state = (1, 1, 1)
        phi = exact_phi(network, state)
        assert math.isfinite(phi)
        assert phi > 0.0

        structure = phi_structure(network, state)
        assert float(structure.phi) == pytest.approx(phi, abs=PHI_TOL)
        assert len(structure.distinctions) > 0
        assert structure.distinctions.sum_phi() > 0.0
        assert len(structure.relations) > 0
        assert structure.relations.sum_phi() > 0.0
        assert math.isfinite(structure.big_phi)

        # The major complex need not be the whole system. Here each mediated
        # pair out-integrates the full triad; largest-first search keeps the
        # first of those tied pairs, (W, S).
        complex_ = maximal_complex(network, state)
        assert float(complex_.phi) > phi
        assert tuple(complex_.node_indices) == (0, 1)

    def test_ternary_factored_system_has_zero_phi(self):
        sbs = deterministic_sbs((3, 3, 3), lambda state: state)
        network = MultivaluedNetwork(
            sbs, (3, 3, 3), cm=np.eye(3), node_labels=("A", "B", "C")
        )
        assert exact_phi(network, (1, 2, 0)) == pytest.approx(0.0, abs=PHI_TOL)

        disconnected = MultivaluedNetwork(
            sbs, (3, 3, 3), cm=np.zeros((3, 3)), node_labels=("A", "B", "C")
        )
        assert exact_phi(disconnected, (0, 0, 0)) == pytest.approx(0.0, abs=PHI_TOL)
        assert float(maximal_complex(disconnected, (0, 0, 0)).phi) == pytest.approx(
            0.0, abs=PHI_TOL
        )

    def test_mixed_radix_phi_structure(self):
        radix = (2, 3, 2)
        sbs = deterministic_sbs(radix, mixed_radix_next)
        assert sbs.shape == (12, 12)
        network = MultivaluedNetwork(
            sbs, radix, cm=TRIAD_CM, node_labels=("A", "B", "C")
        )
        # (1, 1, 1): B' = 1+1 = 2, A' = C' = 1, so the state is reachable
        # from itself only if B' == 1. Use the image (1, 2, 1), which
        # (1, 1, 1) and (1, 2, 1) both... check a fixed point below.
        state = (1, 2, 1)
        # A' = 1 because B>0, B' = 1+1 = 2, C' = 1. Fixed point, reachable.
        phi = exact_phi(network, state)
        assert math.isfinite(phi)
        assert phi > 0.0

        structure = phi_structure(network, state)
        assert float(structure.phi) == pytest.approx(phi, abs=PHI_TOL)
        assert len(structure.distinctions) > 0
        assert structure.distinctions.sum_phi() > 0.0
        assert len(structure.relations) > 0
        assert math.isfinite(structure.sum_phi_relations)
        assert math.isfinite(structure.big_phi)

        complex_ = maximal_complex(network, state)
        assert float(complex_.phi) > 0.0
        assert set(complex_.node_indices) <= {0, 1, 2}
        assert len(complex_.node_indices) >= 2
