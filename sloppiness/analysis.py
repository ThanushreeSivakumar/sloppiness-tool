from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SloppinessResult:
    deltas: np.ndarray
    gamma_min: np.ndarray
    gamma_max: np.ndarray
    psi: np.ndarray
    n_samples: list[int]


@dataclass
class PerParamResult:
    param_index: int
    param_name: str
    deltas: np.ndarray
    gamma_min: np.ndarray
    gamma_max: np.ndarray


def _sample_nball(center: np.ndarray, radius: float,
                  n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = center.shape[0]
    directions = rng.standard_normal((n_samples, n))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms
    u = rng.uniform(0.0, 1.0, size=(n_samples, 1))
    radii = radius * (u ** (1.0 / n))
    samples = center + directions * radii
    return samples


def _compute_gamma(y_ref: np.ndarray, y_sample: np.ndarray) -> float:
    return float(np.sum((y_ref - y_sample) ** 2))


def sloppiness_analysis(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_star: np.ndarray,
    t: np.ndarray,
    delta_max: float = 0.3,
    n_radii: int = 30,
    n0: int = 200,
    alpha: float = 5.0,
    seed: int = 42,
) -> SloppinessResult:

    theta_star = np.asarray(theta_star, dtype=float)
    n_params = theta_star.shape[0]
    rng = np.random.default_rng(seed)

    deltas = np.linspace(delta_max / n_radii, delta_max, n_radii)

    y_ref = model_fn(theta_star, t)

    gamma_min = np.zeros(n_radii)
    gamma_max = np.zeros(n_radii)
    n_samples_list: list[int] = []
    current_n = n0

    for k in range(n_radii):
        n_samples = int(np.ceil(current_n))
        n_samples_list.append(n_samples)

        samples = _sample_nball(theta_star, deltas[k], n_samples, rng)

        gammas = np.empty(n_samples)
        for j in range(n_samples):
            y_j = model_fn(samples[j], t)
            gammas[j] = _compute_gamma(y_ref, y_j)

        gamma_min[k] = gammas.min()
        gamma_max[k] = gammas.max()

        if k < n_radii - 1:
            ratio = deltas[k + 1] / deltas[k]
            current_n = current_n + alpha * (ratio ** n_params)

    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(gamma_max > 0, 1.0 - gamma_min / gamma_max, 0.0)

    return SloppinessResult(
        deltas=deltas,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        psi=psi,
        n_samples=n_samples_list,
    )


def per_parameter_analysis(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_star: np.ndarray,
    t: np.ndarray,
    param_names: list[str] | None = None,
    delta_max: float = 0.3,
    n_radii: int = 30,
    n_samples_per_radius: int = 200,
    seed: int = 42,
) -> list[PerParamResult]:

    theta_star = np.asarray(theta_star, dtype=float)
    n_params = theta_star.shape[0]
    rng = np.random.default_rng(seed)

    if param_names is None:
        param_names = [f"theta{i+1}" for i in range(n_params)]

    y_ref = model_fn(theta_star, t)
    deltas = np.linspace(delta_max / n_radii, delta_max, n_radii)

    results: list[PerParamResult] = []

    for i in range(n_params):
        gmin = np.zeros(n_radii)
        gmax = np.zeros(n_radii)

        for k in range(n_radii):
            perturbations = rng.uniform(-deltas[k], deltas[k],
                                        size=n_samples_per_radius)
            gammas = np.empty(n_samples_per_radius)

            for j in range(n_samples_per_radius):
                theta_pert = theta_star.copy()
                theta_pert[i] += perturbations[j]
                y_j = model_fn(theta_pert, t)
                gammas[j] = _compute_gamma(y_ref, y_j)

            gmin[k] = gammas.min()
            gmax[k] = gammas.max()

        results.append(PerParamResult(
            param_index=i,
            param_name=param_names[i],
            deltas=deltas,
            gamma_min=gmin,
            gamma_max=gmax,
        ))

    return results
