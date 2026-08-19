# Sloppiness Analysis Tool

A minimal Python implementation of the numerical sloppiness detection method from:

> Jagadeesan, Raman & Tangirala (2023), *"Sloppiness: Fundamental study, new
> formalism and its application in model assessment"*, PLoS ONE 18(3): e0282609.

The tool detects **sloppiness**, **local structural unidentifiability**, and
**parameter sensitivity** in dynamical (ODE) models by sampling parameters inside
a growing ball around a reference point and tracking how much the model output
changes.

## Quick Start

```bash
pip install -r requirements.txt
python run_example.py
```

Plots are saved into `outputs/scenario_*/` and textual interpretations are printed
to the console.

## How It Works

### Algorithm (Paper Fig. 14)

1. Choose a reference parameter vector **θ\*** and a maximum perturbation radius
   δ\_max.
2. For each radius δ\_k in a grid [δ\_1, …, δ\_l]:
   - Sample N(k) parameter vectors **uniformly** inside an n-ball of radius δ\_k
     centred at θ\*.
   - Simulate the model at θ\* (reference output y\*) and at each sampled θ.
   - Compute the sum-of-squared error γ(θ) = Σ\_t (y\*(t) − y(t,θ))² for each
     sample.
   - Record γ\_min(δ\_k) and γ\_max(δ\_k).
3. Compute the **model sensitivity index** ψ(δ\_k) = 1 − γ\_min / γ\_max.
4. Also run a **per-parameter** version: vary each θ\_i individually while holding
   all other parameters fixed.

### Test Model

Two-state linear ODE (Paper Eq. 29):

```
dx₁/dt = −θ₁·x₁,   dx₂/dt = −θ₂·x₂
y(t)   = x₁(t) + x₂(t) = e^{−θ₁t} + e^{−θ₂t}
x₁(0)  = x₂(0) = 1
```

Three scenarios from Table 3 (all with δ\_max = 0.3):

| Scenario | θ\*         | Expected behaviour           |
|----------|-------------|------------------------------|
| 1        | [0.4, 1.0]  | Non-sloppy                   |
| 2        | [1.0, 10.0] | Sloppy                       |
| 3        | [0.4, 0.5]  | Locally structurally unident.|

## How to Interpret the Plots

### γ vs δ (sloppiness plot)

- **γ\_min grows with δ** → non-sloppy: every parameter perturbation is
  distinguishable from θ\*.
- **γ\_min stays near 0 while γ\_max grows** → sloppy: some parameter directions
  produce nearly identical outputs.
- **γ\_min ≈ 0 exactly** → locally structurally unidentifiable: there exist
  *infinitely many* parameter sets giving the *exact same* output.

### ψ vs δ (model sensitivity index)

- **ψ ≈ 1** → highly anisotropic sensitivity (sloppy); the model is much more
  sensitive in some parameter directions than others.
- **ψ ≪ 1** → roughly isotropic sensitivity (well-conditioned); perturbations
  in all directions have comparable effects.

### Per-parameter plots

Show which individual parameters are sensitive (γ grows when perturbed) vs
insensitive (γ stays low). Helps identify which parameters are responsible for
sloppiness.

## Project Structure

```
sloppiness-tool/
├── README.md
├── requirements.txt
├── run_example.py              # Run all 3 scenarios
├── sloppiness/
│   ├── __init__.py
│   ├── model.py                # ODE model definition + simulate()
│   ├── analysis.py             # n-ball sampling, γ_min/γ_max, ψ
│   ├── plotting.py             # Matplotlib plotting functions
│   └── interpret.py            # Rule-based text interpretation
└── outputs/
    ├── scenario_1_non_sloppy/
    ├── scenario_2_sloppy/
    └── scenario_3_unidentifiable/
```

## Extending to Other Models

The analysis code is model-agnostic. To use a different ODE model, create a
callable with the signature:

```python
def my_model(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return y(t) given parameters theta and time grid t."""
    ...
```

Then pass it to `sloppiness_analysis()` and `per_parameter_analysis()` in place
of `TwoStateLinearModel()`.

## Dependencies

- numpy ≥ 1.24
- scipy ≥ 1.10
- matplotlib ≥ 3.7

## Reference

Jagadeesan P, Raman K, Tangirala AK (2023) Sloppiness: Fundamental study, new
formalism and its application in model assessment. PLoS ONE 18(3): e0282609.
https://doi.org/10.1371/journal.pone.0282609.
