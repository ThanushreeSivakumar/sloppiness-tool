from __future__ import annotations

import numpy as np

def simulate(model_fn, theta: np.ndarray, t: np.ndarray) -> np.ndarray:

    y = model_fn(theta, t)
    assert y.shape == t.shape, (
        f"Model output shape {y.shape} must match time grid shape {t.shape}"
    )
    return y

class TwoStateLinearModel:

    def __init__(self, x0: tuple[float, float] = (1.0, 1.0)):
        self.x0 = x0
        self.n_params = 2  

    def __call__(self, theta: np.ndarray, t: np.ndarray) -> np.ndarray:

        theta = np.asarray(theta, dtype=float)
        assert theta.shape == (2,), f"Expected 2 parameters, got {theta.shape}"
        x1_0, x2_0 = self.x0
        y = x1_0 * np.exp(-theta[0] * t) + x2_0 * np.exp(-theta[1] * t)
        return y
