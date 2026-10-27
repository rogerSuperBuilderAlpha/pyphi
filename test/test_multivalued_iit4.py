"""Opt-in multivalued IIT 4.0: state space, binary parity, and Φ-structure."""

import math

import numpy as np
import pytest

from pyphi import Network, Subsystem, convert, new_big_phi
from pyphi.conf import config
from pyphi.exceptions import ConditionallyDependentError
from pyphi.direction import Direction
from pyphi.exceptions import StateUnreachableError
from pyphi.multivalued import (
    MultivaluedExplicitTPM,
    MultivaluedNetwork,
    MultivaluedSubsystem,
    exact_phi,
    maximal_complex,
    phi_structure,
    sia,
)
from pyphi.multivalued.conditional_independence import encode_state
from pyphi.multivalued.utils_mv import all_states_mixed
from pyphi.tpm import ExplicitTPM
from pyphi.utils import all_states, powerset

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

    def test_cm_must_be_square(self):
        sbs = deterministic_sbs((3, 3), lambda state: (state[1], state[0]))
        with pytest.raises(ValueError):
            MultivaluedNetwork(sbs, (3, 3), cm=np.ones((2, 3)))
        with pytest.raises(ValueError):
            MultivaluedNetwork(sbs, (3, 3), cm=np.ones((3, 3)))

    def test_network_equality_and_hash(self):
        sbs = deterministic_sbs((3, 3), lambda state: (state[1], state[0]))
        net_a = MultivaluedNetwork(sbs, (3, 3), node_labels=("A", "B"))
        net_b = MultivaluedNetwork(sbs, (3, 3), node_labels=("A", "B"))
        assert net_a == net_b
        assert not (net_a != net_b)
        assert hash(net_a) == hash(net_b)
        assert len({net_a, net_b}) == 1

        other_cm = MultivaluedNetwork(sbs, (3, 3), cm=np.eye(2))
        assert net_a != other_cm

        other_tpm = MultivaluedNetwork(
            deterministic_sbs((3, 3), lambda state: state), (3, 3)
        )
        assert net_a != other_tpm

        # Same matrix read with a different alphabet is a different network.
        binary = deterministic_sbs((2, 2), lambda state: (state[1], state[0]))
        stock = Network(convert.state_by_state2state_by_node(binary))
        assert mv_network_from_sbn(stock.tpm.tpm, stock.cm, ("A", "B"))[0] != stock
        assert stock != mv_network_from_sbn(stock.tpm.tpm, stock.cm, ("A", "B"))[0]

    def test_all_states_mixed_is_little_endian(self):
        assert list(all_states_mixed((2, 2))) == list(all_states(2))
        assert list(all_states_mixed((2, 2, 2))) == list(all_states(3))
        states = list(all_states_mixed((2, 3)))
        assert states == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
        assert [encode_state(s, (2, 3)) for s in states] == list(range(6))


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


def random_sbn(rng, n, noisy):
    """Random state-by-node TPM: deterministic (0/1) or noisy (uniform)."""
    if noisy:
        return rng.uniform(0.0, 1.0, size=(2**n, n))
    return rng.integers(0, 2, size=(2**n, n)).astype(float)


def cm_from_sbn(sbn):
    """Connectivity matrix with an edge j -> i iff node i depends on node j."""
    n = sbn.shape[-1]
    md = convert.to_multidimensional(sbn)
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            off = np.take(md[..., i], 0, axis=j)
            on = np.take(md[..., i], 1, axis=j)
            if not np.allclose(off, on):
                cm[j, i] = 1
    return cm


def complex_summary(result):
    """(Φ, node indices) of a complex; NullPhiStructure has no indices."""
    return float(result.phi), tuple(getattr(result, "node_indices", None) or ())


def assert_binary_parity(seed, n, noisy, tol=1e-9):
    """Radix-2 multivalued path must match stock pyphi on a random network.

    For every network state and every non-empty subsystem, compares
    reachability, system Φ, the major complex, and the cause and effect
    repertoires over every (mechanism, purview) pair.
    """
    rng = np.random.default_rng(seed)
    sbn = random_sbn(rng, n, noisy)
    cm = cm_from_sbn(sbn)
    labels = tuple("ABCD"[:n])
    stock_network = Network(sbn, cm=cm, node_labels=labels)
    network, _sbs = mv_network_from_sbn(sbn, cm, labels)

    for state in all_states(n):
        stock_complex = new_big_phi.maximal_complex(stock_network, state)
        our_complex = maximal_complex(network, state)
        stock_phi, stock_nodes = complex_summary(stock_complex)
        our_phi, our_nodes = complex_summary(our_complex)
        assert our_phi == pytest.approx(stock_phi, abs=tol), (seed, state)
        assert our_nodes == stock_nodes, (seed, state)

        for nodes in powerset(range(n), nonempty=True):
            try:
                stock_subsystem = Subsystem(stock_network, state, nodes)
            except StateUnreachableError:
                with pytest.raises(StateUnreachableError):
                    MultivaluedSubsystem(network, state, nodes)
                continue
            subsystem = MultivaluedSubsystem(network, state, nodes)

            stock_sia = new_big_phi.sia(stock_subsystem)
            our_sia = sia(subsystem)
            our_phi, stock_phi = float(our_sia.phi), float(stock_sia.phi)
            assert our_phi == pytest.approx(stock_phi, abs=tol), (seed, state, nodes)

            for mechanism in powerset(nodes, nonempty=True):
                for purview in powerset(nodes, nonempty=True):
                    for direction in Direction.both():
                        expected = stock_subsystem.repertoire(
                            direction, mechanism, purview
                        )
                        actual = subsystem.repertoire(direction, mechanism, purview)
                        assert actual.shape == expected.shape
                        assert np.allclose(actual, expected, atol=tol, rtol=0.0), (
                            seed,
                            state,
                            nodes,
                            mechanism,
                            purview,
                            direction,
                        )


class TestRandomBinaryParity:
    """Seeded random binary networks, every state, every subsystem."""

    @pytest.mark.parametrize(
        "seed,noisy",
        [(1, False), (2, False), (3, False), (4, True), (5, True), (6, True)],
    )
    def test_three_nodes(self, seed, noisy):
        assert_binary_parity(seed, 3, noisy)

    @pytest.mark.slow
    @pytest.mark.parametrize("seed,noisy", [(7, False), (8, True)])
    def test_four_nodes(self, seed, noisy):
        assert_binary_parity(seed, 4, noisy)


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

        # The major complex need not be the whole system. The rule and the
        # state are symmetric under W <-> C, so the pairs (W, S) and (S, C)
        # tie, and each out-integrates the full triad. The search visits
        # subsystems in stock order (``utils.powerset(..., reverse=True)``),
        # so (S, C) is seen first and, as in stock ``maximal_complex``, the
        # first subsystem at the maximum wins.
        phi_ws = float(sia(MultivaluedSubsystem(network, state, (0, 1))).phi)
        phi_sc = float(sia(MultivaluedSubsystem(network, state, (1, 2))).phi)
        assert phi_ws == pytest.approx(phi_sc, abs=PHI_TOL)
        assert phi_sc > phi
        complex_ = maximal_complex(network, state)
        assert float(complex_.phi) == pytest.approx(phi_sc, abs=PHI_TOL)
        assert tuple(complex_.node_indices) == (1, 2)

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
        # (1, 2, 1) is a fixed point: A' = C' = 1 because B > 0, and
        # B' = A + C = 2. It is therefore reachable (from itself).
        state = (1, 2, 1)
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
