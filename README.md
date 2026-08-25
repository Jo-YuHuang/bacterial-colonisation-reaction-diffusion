# Bacterial Colonisation: Reaction–Diffusion Simulation

A Python numerical simulation of bacterial colonisation in a 10 cm Petri-dish domain. The model compares an explicit forward-time, central-space solver with a semi-implicit solver across multiple grid resolutions, then visualises coverage, temporal growth, and final bacterial density.

## What this project demonstrates

- Discretisation of coupled reaction–diffusion partial differential equations.
- Five-point finite-difference Laplacian with no-flux (Neumann) boundaries.
- Explicit time integration with a grid-dependent stability time step.
- Semi-implicit diffusion using sparse matrices and SciPy's sparse linear solver.
- Grid-resolution and convergence comparison from 50×50 to 400×400 nodes.
- Quantitative coverage metrics and density heatmap visualisation.
- Reproducible initial conditions through a fixed random seed.

## Model

The simulation tracks two fields over a square 10 cm domain:

- u: glucose concentration
- v: bacterial concentration

The coupled reaction–diffusion system is represented by:

    ∂u/∂t = Du∇²u − uv² + F(1 − u)
    ∂v/∂t = Dv∇²v + uv² − (F + k)v

The initial bacterial population is a centred circular seed with a 1 cm physical radius. A small perturbation is applied to the initial concentration fields to avoid an artificially symmetric solution.

## Numerical methods

### Explicit solver

The explicit solver uses forward Euler time integration and a central finite-difference spatial discretisation. Its time step scales with the square of the grid spacing to satisfy the diffusion stability condition.

### Semi-implicit solver

The semi-implicit solver treats the diffusion term through a sparse backward-Euler matrix solve while evaluating reaction terms explicitly. The sparse system is assembled for each grid resolution and solved with SciPy's sparse solver.

### Boundary conditions

No-flux Neumann boundary conditions are implemented directly in the Laplacian stencil. Edge and corner points use reflected-neighbour contributions so material does not flow through the Petri-dish boundary.

## Outputs

The script produces three plots:

1. Final bacterial coverage after 10 simulated hours versus grid size for both solvers.
2. Explicit bacterial coverage over time at 100×100 nodes.
3. A bacterial-density heatmap after 10 simulated hours at 400×400 nodes.

During execution, the script also reports final coverage and wall-clock runtime for each grid resolution.

## Repository structure

    src/
    ├── reaction_diffusion_simulation.py  corrected portfolio version
    └── simulation_original.py             uploaded reference version
    requirements.txt
    README.md

## Run locally

Create an environment and install the dependencies:

    python -m venv .venv
    .venv\Scripts\activate          # Windows PowerShell
    source .venv/bin/activate          # macOS/Linux
    pip install -r requirements.txt

Run the simulation:

    python src/reaction_diffusion_simulation.py

The 400×400 semi-implicit case is computationally heavier than the lower-resolution cases. For a quick test, reduce N_values in the script before running the full resolution study.

## Interpretation and limitations

The grid-resolution comparison is intended to assess numerical behaviour and computational cost, not to provide a validated biological prediction. The parameters, initial conditions, and threshold for bacterial presence are modelling assumptions.

Potential next improvements include:

- Separating configuration, solver functions, and plotting into modules.
- Adding automated checks for boundary stencils and conservation behaviour.
- Recording results to CSV for reproducible comparison.
- Benchmarking sparse solver performance and memory use.
- Adding a command-line interface for resolution, time horizon, and output directory.
- Comparing the numerical solution against a manufactured or analytical test case.

## Portfolio context

This project demonstrates numerical modelling, scientific Python, algorithmic reasoning, computational performance awareness, and clear communication of simulation results.
