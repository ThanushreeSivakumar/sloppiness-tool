from __future__ import annotations

import numpy as np

from .analysis import SloppinessResult, PerParamResult


def interpret_scenario(
    result: SloppinessResult,
    scenario_name: str = "Scenario",
    near_zero_abs: float = 1e-6,
) -> str:

    gmin_final = result.gamma_min[-1]
    gmax_final = result.gamma_max[-1]
    psi_final = result.psi[-1]
    delta_final = result.deltas[-1]

    outer_half = len(result.psi) // 2
    psi_mean = float(np.mean(result.psi[outer_half:]))

    n_zero = int(np.sum(result.gamma_min < near_zero_abs))
    frac_zero = n_zero / len(result.gamma_min)

    gmin_outer = result.gamma_min[outer_half:]
    gmin_mid = result.gamma_min[outer_half]
    if gmin_mid > 0:
        growth_ratio = gmin_final / gmin_mid
    else:
        growth_ratio = 0.0

    ratio_at_final = gmin_final / gmax_final if gmax_final > 0 else 0.0

    lines: list[str] = [f"{'='*60}", f"  {scenario_name}", f"{'='*60}"]

    if frac_zero > 0.5:
        classification = "LOCALLY STRUCTURALLY UNIDENTIFIABLE"
        lines.append(f"  Classification: {classification}")
        lines.append(
            f"  gamma_min ~ 0 (< {near_zero_abs:.1e}) at "
            f"{n_zero}/{len(result.gamma_min)} radii, indicating that "
            f"there exist parameter combinations producing *identical* "
            f"model output."
        )
    elif ratio_at_final > 0.05:
        classification = "NON-SLOPPY"
        lines.append(f"  Classification: {classification}")
        lines.append(
            f"  At delta = {delta_final:.4f}: gamma_min = {gmin_final:.4e}, "
            f"gamma_max = {gmax_final:.4e} "
            f"(ratio = {ratio_at_final:.4e})."
        )
        lines.append(
            f"  gamma_min grows with delta -- all parameter "
            f"perturbations produce distinguishable model outputs."
        )
    else:
        classification = "SLOPPY"
        lines.append(f"  Classification: {classification}")
        lines.append(
            f"  At delta = {delta_final:.4f}: gamma_min = {gmin_final:.4e}, "
            f"gamma_max = {gmax_final:.4e} "
            f"(ratio = {ratio_at_final:.4e})."
        )
        lines.append(
            f"  gamma_min remains very small compared to gamma_max, meaning "
            f"large regions of parameter space yield nearly identical outputs."
        )

    lines.append(
        f"  psi (mean, outer half): {psi_mean:.4f}  |  "
        f"psi at delta_max: {psi_final:.4f}"
    )
    if psi_mean > 0.95:
        lines.append(
            "  -> psi ~ 1: highly anisotropic sensitivity (sloppy)."
        )
    elif psi_mean < 0.5:
        lines.append(
            "  -> psi well below 1: roughly isotropic sensitivity "
            "(well-conditioned)."
        )
    else:
        lines.append(
            "  -> psi moderately high: some directional sensitivity "
            "imbalance."
        )

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def interpret_per_parameter(
    results: list[PerParamResult],
    scenario_name: str = "Scenario",
) -> str:

    lines = [f"  Per-parameter sensitivity for {scenario_name}:"]

    for r in results:
        gmax_final = r.gamma_max[-1]
        gmin_final = r.gamma_min[-1]
        delta_final = r.deltas[-1]

        if gmax_final < 1e-8:
            label = "INSENSITIVE (gamma_max ~ 0)"
        elif gmax_final > 0.1:
            label = "SENSITIVE (high gamma_max)"
        elif gmax_final > 0.001:
            label = "MODERATELY SENSITIVE"
        else:
            label = "LOW SENSITIVITY"

        lines.append(
            f"    {r.param_name}: gamma_max={gmax_final:.4e}, "
            f"gamma_min={gmin_final:.4e} at delta={delta_final:.3f} "
            f"-> {label}"
        )

    return "\n".join(lines)
