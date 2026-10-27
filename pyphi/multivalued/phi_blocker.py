"""Probe exact IIT-4.0 Φ on a MultivaluedNetwork (M2)."""

from __future__ import annotations

import traceback
from typing import Any

from .network import MultivaluedNetwork
from .sia_mv import exact_phi
from .tpm import MultivaluedTPMError


def probe_exact_phi_blocker(
    network: MultivaluedNetwork, state: tuple[int, ...]
) -> dict[str, Any]:
    """Attempt exact Φ; report success or remaining blocker locus."""
    report: dict[str, Any] = {
        "attempted": True,
        "phi": None,
        "blocked": True,
        "locus": "",
        "error_type": "",
        "error": "",
        "traceback_tail": "",
    }
    try:
        phi = exact_phi(network, state)
        report["blocked"] = False
        report["phi"] = float(phi)
        report["locus"] = "phi_computed"
    except MultivaluedTPMError as e:
        report["error_type"] = type(e).__name__
        report["error"] = str(e)
        report["locus"] = str(e)
    except Exception as e:  # noqa: BLE001
        report["error_type"] = type(e).__name__
        report["error"] = str(e)
        report["locus"] = f"unexpected: {type(e).__name__}"
        report["traceback_tail"] = "".join(
            traceback.format_exception(type(e), e, e.__traceback__)[-4:]
        )
    return report
