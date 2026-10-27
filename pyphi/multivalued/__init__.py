# -*- coding: utf-8 -*-
"""Opt-in multivalued (mixed-radix) IIT 4.0.

Stock ``ExplicitTPM`` / ``Subsystem`` index states as binary: a square
state-by-state TPM with ``S`` columns is treated as ``N = int(log2(S))``
units, and repertoire arrays are shaped ``[2] * N``. This subpackage does not
replace that path. It accepts an explicit alphabet ``(k_0, ..., k_{n-1})``
with each ``k_i >= 2``, keeps the TPM in state-by-state form, and computes
cause/effect repertoires, intrinsic information (GID), system ``Φ_s``,
distinctions, relations, and major complexes on that alphabet.

Binary networks (``k_i = 2`` for every node) match ``pyphi.new_big_phi`` on
the IIT 4.0 feature line.

Example::

    from pyphi.multivalued import (
        MultivaluedNetwork,
        exact_phi,
        phi_structure,
        maximal_complex,
    )
"""

from .complexes import maximal_complex
from .network import MultivaluedNetwork
from .phi_blocker import probe_exact_phi_blocker
from .sia_mv import exact_phi, phi_structure, sia
from .subsystem_mv import MultivaluedSubsystem
from .tpm import MultivaluedExplicitTPM, MultivaluedTPMError, SBSNativeExplicitTPM

__all__ = [
    'MultivaluedNetwork',
    'MultivaluedExplicitTPM',
    'SBSNativeExplicitTPM',
    'MultivaluedTPMError',
    'MultivaluedSubsystem',
    'exact_phi',
    'sia',
    'phi_structure',
    'maximal_complex',
    'probe_exact_phi_blocker',
]
