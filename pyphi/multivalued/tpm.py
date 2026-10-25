# -*- coding: utf-8 -*-
"""SBS-native ExplicitTPM for multivalued (mixed-radix) alphabets.

Keeps the TPM in square state-by-state form. Never calls PyPhi's binary
``state_by_state2state_by_node`` / ``int(log2(...))`` path.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from pyphi.exceptions import ConditionallyDependentError
from pyphi.utils import np_hash, np_immutable

from .conditional_independence import sbs_conditionally_independent


class MultivaluedTPMError(RuntimeError):
    """Raised when an IIT-4.0 call needs work past M1 (SBS ingest)."""


class MultivaluedExplicitTPM:
    """Square state-by-state TPM with ``num_states_per_node``.

    Binary restriction (all alphabets = 2) is allowed and still stays SBS;
    it is not converted to multidimensional state-by-node form.
    """

    def __init__(
        self,
        tpm,
        num_states_per_node: Sequence[int],
        validate: bool = True,
        check_independence: bool = True,
    ):
        self._num_states_per_node = tuple(int(k) for k in num_states_per_node)
        if not self._num_states_per_node:
            raise ValueError('num_states_per_node must be non-empty')
        if any(k < 2 for k in self._num_states_per_node):
            raise ValueError(
                f"each alphabet size must be >= 2; got {self._num_states_per_node}"
            )

        arr = np.array(tpm, dtype=float)
        expected = int(np.prod(self._num_states_per_node))
        if arr.ndim != 2 or arr.shape != (expected, expected):
            raise ValueError(
                f"Invalid SBS shape {arr.shape}; expected "
                f"{(expected, expected)} for num_states_per_node="
                f"{self._num_states_per_node}"
            )

        self._tpm = arr
        if validate:
            self.validate(check_independence=check_independence)

        self._tpm = np_immutable(self._tpm)
        self._hash = np_hash(self._tpm)

    # --- Array-like surface used by callers / tests ---

    @property
    def tpm(self) -> np.ndarray:
        return self._tpm

    @property
    def shape(self):
        return self._tpm.shape

    @property
    def ndim(self) -> int:
        return self._tpm.ndim

    @property
    def num_states_per_node(self) -> tuple[int, ...]:
        return self._num_states_per_node

    @property
    def number_of_units(self) -> int:
        return len(self._num_states_per_node)

    @property
    def num_states(self) -> int:
        return int(np.prod(self._num_states_per_node))

    def is_state_by_state(self) -> bool:
        return True

    def is_binary(self) -> bool:
        return all(k == 2 for k in self._num_states_per_node)

    def is_deterministic(self) -> bool:
        return bool(np.all(np.logical_or(self._tpm == 0.0, self._tpm == 1.0)))

    def validate(self, check_independence: bool = True) -> bool:
        if (self._tpm < 0.0).any() or (self._tpm > 1.0).any():
            raise ValueError(
                'Invalid TPM: probabilities must be in the interval [0, 1].'
            )
        if not np.allclose(np.sum(self._tpm, axis=1), 1.0, atol=1e-12):
            raise ValueError('Invalid TPM: probabilities must sum to 1.')
        if check_independence:
            self.conditionally_independent()
        return True

    def conditionally_independent(self) -> bool:
        """Mixed-radix CI — product of per-node categorical marginals.

        Does **not** use binary SBN round-trip (the CI-off ``log2`` trap).
        """
        if not sbs_conditionally_independent(
            self._tpm, self._num_states_per_node
        ):
            raise ConditionallyDependentError(
                'TPM is not conditionally independent under mixed-radix '
                'categorical factorization (SBS-native check; no binary SBN).'
            )
        return True

    def to_multidimensional_state_by_node(self):
        raise MultivaluedTPMError(
            'SBS-native M1 TPM has no binary multidimensional SBN form. '
            'M2 must implement categorical / mixed-radix conditioning and '
            'repertoires without int(log2) conversion.'
        )

    def condition_tpm(self, condition: Mapping[int, int]):
        # Empty background (full-system Subsystem): still SBS-native; no
        # conversion. Non-empty conditioning is M2.
        if not condition:
            return self
        raise MultivaluedTPMError(
            'M1 blocker past ExplicitTPM: non-empty condition_tpm requires '
            'mixed-radix SBS (or categorical) conditioning. Stock IIT-4.0 '
            'Subsystem.condition_tpm assumes multidimensional binary SBN '
            f"layout; refused condition={dict(condition)}."
        )

    def sbs_copy(self) -> np.ndarray:
        """Return a mutable copy of the preserved SBS matrix."""
        return np.array(self._tpm, copy=True)

    def __array__(self, dtype=None):
        return np.asarray(self._tpm, dtype=dtype)

    def __repr__(self) -> str:
        return (
            f"MultivaluedExplicitTPM(shape={self.shape}, "
            f"num_states_per_node={self._num_states_per_node})"
        )


# Name used by the original vendored pin.
SBSNativeExplicitTPM = MultivaluedExplicitTPM
