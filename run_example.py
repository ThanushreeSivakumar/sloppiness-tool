from __future__ import annotations

import numpy as np

from sloppiness.model import TwoStateLinearModel
from sloppiness.analysis import sloppiness_analysis, per_parameter_analysis
from sloppiness.plotting import (
    plot_gamma,
    plot_psi,
    plot_per_parameter,
    plot_per_parameter_combined,
)
from sloppiness.interpret import interpret_scenario, interpret_per_parameter

SCENARIOS = {
    "Scenario 1 - Non-sloppy": {
        "theta_star": np.array([0.4, 1.0]),
        "outdir": "outputs/scenario_1_non_sloppy",
    },
    "Scenario 2 - Sloppy": {
        "theta_star": np.array([1.0, 10.0]),
        "outdir": "outputs/scenario_2_sloppy",
    },
    "Scenario 3 - Unidentifiable": {
        "theta_star": np.array([0.4, 0.5]),
        "outdir": "outputs/scenario_3_unidentifiable",
    },
}

T_GRID = np.linspace(0, 5, 50)     
DELTA_MAX = 0.3                     
N_RADII = 40                        
N0 = 500                            
ALPHA = 5.0                         
PARAM_NAMES = ["theta1", "theta2"]

def main():
    model = TwoStateLinearModel(x0=(1.0, 1.0))

    for name, cfg in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"  theta* = {cfg['theta_star']}")
        print(f"{'='*60}")

        result = sloppiness_analysis(
            model_fn=model,
            theta_star=cfg["theta_star"],
            t=T_GRID,
            delta_max=DELTA_MAX,
            n_radii=N_RADII,
            n0=N0,
            alpha=ALPHA,
        )

        plot_gamma(result, title=f"{name} — γ vs δ", outdir=cfg["outdir"])
        plot_psi(result, title=f"{name} — ψ vs δ", outdir=cfg["outdir"])

        pp_results = per_parameter_analysis(
            model_fn=model,
            theta_star=cfg["theta_star"],
            t=T_GRID,
            param_names=PARAM_NAMES,
            delta_max=DELTA_MAX,
            n_radii=N_RADII,
        )

        plot_per_parameter(pp_results, title_prefix=f"{name} — ",
                           outdir=cfg["outdir"])
        plot_per_parameter_combined(pp_results, title=f"{name} — Per-Param",
                                    outdir=cfg["outdir"])

        diag = interpret_scenario(result, scenario_name=name)
        print(diag)
        pp_diag = interpret_per_parameter(pp_results, scenario_name=name)
        print(pp_diag)

    print("\nDone! All scenarios complete. Check the outputs/ directory for plots.")

if __name__ == "__main__":
    main()
