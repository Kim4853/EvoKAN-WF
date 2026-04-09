# EvoKAN-WF

Weak-form and strong-form evolutionary Kolmogorov-Arnold networks for time-dependent PDEs.

This repository contains notebook-based implementations of evolutionary KAN solvers for:

- the 1D Allen-Cahn equation
- the 2D reaction-diffusion heat equation
- both strong-form and weak-form formulations

The code in this repository is organized as research notebooks rather than a standalone Python package. Each notebook contains the full workflow for:

- defining a KAN model
- fitting the initial condition
- assembling Jacobian-based linear systems
- evolving the trainable parameters in time
- visualizing the predicted solution
- exporting selected diagnostics

## Repository Overview

The four main notebooks are:

| Notebook | PDE | Formulation | Summary |
| --- | --- | --- | --- |
| `1D_Allen-Cahn_EvoKAN-SF.ipynb` | 1D Allen-Cahn | Strong form | Pointwise residual-based evolutionary KAN solver |
| `1D_Allen-Cahn_EvoKAN_WF.ipynb` | 1D Allen-Cahn | Weak form | Weak-form projection using quadrature and test functions |
| `2D_Heat_EvoKAN_SF.ipynb` | 2D heat / reaction-diffusion | Strong form | Pointwise residual solve on a 2D spatial grid |
| `2D_Heat_EvoKAN_WF.ipynb` | 2D heat / reaction-diffusion | Weak form | Projected weak-form solve with tensor-product test functions |

An additional notebook is included:

- `2D_Heat_EvoKAN_WF_2.ipynb`

This file is kept as a reference snapshot of the 2D weak-form workflow. The main notebook intended for use is `2D_Heat_EvoKAN_WF.ipynb`.

The repository also includes:

- `Weak_form_evolutionary_KAN (6).pdf`: reference paper used to align the notebook experiments
- `trained_model/`: saved checkpoints used by some notebook workflows
- `results/`: exported figures and example artifacts

## PDEs Implemented

### 1D Allen-Cahn Equation

The Allen-Cahn notebooks solve:

$$
\partial_t u = \partial_{xx}u - \frac{1}{\varepsilon^2}u(u^2-1), \qquad x \in [-1,1]
$$

with:

$$
u(x,0)=0.08\sin(\pi x), \qquad u(-1,t)=u(1,t)=0
$$

The main parameter used in the notebooks is:

$$
\varepsilon = 0.002
$$

### 2D Heat Equation with Nonlinear Forcing Term

The 2D heat notebooks solve:

$$
\partial_t u = \alpha (u_{xx}+u_{yy}) + u(1-u), \qquad (x,y)\in(-1,1)^2
$$

with:

$$
u(x,y,0)=\cos(\pi x)\cos(\pi y)
$$

and homogeneous Neumann boundary condition:

$$
\frac{\partial u}{\partial n}=0
$$

The diffusion parameter used in the notebooks is:

$$
\alpha = 0.1
$$

## Method Summary

### 1. Initial-condition fitting

Each notebook first trains a KAN to match the prescribed initial condition.

This stage provides:

- a consistent starting field
- an initialized parameter vector
- the baseline state for the evolutionary update

### 2. Parameter-space evolution

Instead of evolving nodal solution values directly, the notebooks evolve the trainable KAN parameters.

If the parameter vector is denoted by $w$, the method computes a linearized update in parameter space using Jacobian information:

$$
J = \frac{\partial u}{\partial w}
$$

The resulting update is solved by least-squares and applied explicitly to the parameter vector.

### 3. Strong-form formulation

In the strong-form notebooks, the PDE residual is enforced pointwise on sampled spatial collocation points.

This means:

- the residual is evaluated directly from automatic differentiation
- the Jacobian maps parameter perturbations to pointwise field perturbations
- the update is obtained by a pointwise linearized solve

### 4. Weak-form formulation

In the weak-form notebooks, the spatial operator is projected onto test functions.

This reduces direct pointwise enforcement of the PDE and replaces it by integral constraints.

For the 1D Allen-Cahn case, the weak form uses quadrature and sine-based test functions.  
For the 2D heat case, the weak form uses tensor-product polynomial test functions together with Gauss-Legendre quadrature.

## Notebook-by-Notebook Details

### `1D_Allen-Cahn_EvoKAN-SF.ipynb`

Core configuration:

- model: `KAN([1, 3, 3, 3, 3, 1])`
- initial-fit optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- initial-fit epochs: `5001`
- time step: `delta_t = 1e-7`
- number of steps: `500`
- linear solver: `scipy.sparse.linalg.lsmr`

Implementation highlights:

- automatic differentiation for $u$, $u_x$, and $u_{xx}$
- strong-form residual evaluation
- energy-density computation
- condition-number sampling
- elapsed-time reporting and per-step statistics

### `1D_Allen-Cahn_EvoKAN_WF.ipynb`

Core configuration:

- model: `KAN([1, 3, 3, 3, 3, 1])`
- initial-fit optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- initial-fit epochs: `5001`
- time step: `delta_t = 1e-7`
- number of steps: `500`
- quadrature nodes: `N_quad = 64`
- test space size: `K = 150`

Implementation highlights:

- strong-form Jacobian evaluated at quadrature nodes
- weak-form projection of the Jacobian
- Fourier sine test functions satisfying the Dirichlet boundary condition
- condition-number monitoring of the projected system

### `2D_Heat_EvoKAN_SF.ipynb`

Core configuration:

- model: `KAN([2, 4, 4, 4, 1])`
- initial-fit grid: `dense_points = 40`, `sparse_points = 0`
- initial-fit optimizer: `AdamW(lr=5e-4, weight_decay=1e-4)`
- initial-fit epochs: `20001`
- time step: `delta_t = 1e-3`
- number of steps: `500`

Implementation highlights:

- 2D autograd for $u_x$, $u_{xx}$, $u_y$, and $u_{yy}$
- pointwise residual solve in strong form
- Neumann-boundary diagnostics
- condition-number sampling and LSMR iteration tracking

### `2D_Heat_EvoKAN_WF.ipynb`

Core configuration:

- model: `KAN([2, 4, 4, 4, 1])`
- initial-fit grid: `dense_points = 64`, `sparse_points = 0`
- initial-fit optimizer: `AdamW(lr=1e-5, weight_decay=1e-4)`
- initial-fit epochs: `30001`
- time step: `delta_t = 1e-3`
- number of steps: `500`
- test function family: `cheb3`
- tensor-product basis counts: `Kx = 15`, `Ky = 15`
- quadrature order: `Nx = 20`, `Ny = 20`

Implementation highlights:

- construction of a pointwise Jacobian at quadrature nodes
- projection to a weak-form matrix
- tensor-product test basis $v_{p,q}(x,y)=V_p(x)V_q(y)$
- projected nonlinear right-hand side
- LSMR-based least-squares solve for the parameter update

## KAN Architecture Used Here

The notebooks implement a notebook-local KAN variant rather than importing an external package.

The model is built from:

- radial basis expansions
- linear mixing layers
- optional base updates
- narrow fully connected layer stacks

The exact implementation differs slightly between the 1D and 2D notebooks, but the workflow is the same:

1. map spatial coordinates to KAN features
2. produce the field value $u$
3. differentiate the model output with respect to both input coordinates and trainable parameters
4. use those derivatives to assemble a time-evolution step in parameter space

## Dependencies

The notebooks use the following Python packages:

- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `jupyter` or `notebook`

Optional GPU-related imports appear in some notebooks:

- `cupy`
- `cupyx`

Some code paths are CPU-based even when GPU-related packages are imported, so GPU support is helpful but not strictly required for every notebook cell.

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use CuPy-backed GPU paths, install the CuPy build that matches your CUDA version separately.

Example:

```bash
pip install cupy-cuda11x
```

## How To Run

Open the notebooks in Jupyter:

```bash
jupyter notebook
```

Recommended execution pattern:

1. Open one notebook at a time.
2. Run the cells from top to bottom.
3. Fit the initial condition first.
4. Run the time-marching cells.
5. Run the visualization or export cells last.

Suggested order for first-time inspection:

1. `1D_Allen-Cahn_EvoKAN-SF.ipynb`
2. `1D_Allen-Cahn_EvoKAN_WF.ipynb`
3. `2D_Heat_EvoKAN_SF.ipynb`
4. `2D_Heat_EvoKAN_WF.ipynb`

## Outputs and Artifacts

### `trained_model/`

Contains saved checkpoints produced by notebook runs.  
These are mainly intermediate or exported parameter snapshots.

### `results/`

Contains exported figures and result summaries.  
Additional notebook runs may generate more figures or diagnostic files.

### Notebook-generated outputs

Depending on the notebook, you may see:

- contour plots
- solution profiles
- checkpoint files
- condition-number logs
- LSMR iteration statistics
- boundary-gradient diagnostics

## Reproducibility Notes

- The notebooks set random seeds in several places, but exact reproducibility can still depend on hardware, library versions, and solver tolerances.
- The notebooks are research workflows, not hardened production solvers.
- Increasing LSMR iteration counts does not necessarily improve the solution. In these evolutionary updates, overly accurate solves can amplify ill-conditioning and linearization error.
- The weak-form and strong-form notebooks are not intended to be numerically identical. They solve the same PDEs using different residual enforcement strategies.

## Important Usage Notes

- These notebooks are the primary source of truth in this repository.
- The code is intentionally notebook-centric.
- Some cells save checkpoints or export data files into local folders.
- `2D_Heat_EvoKAN_WF_2.ipynb` is retained as a reference notebook and may be useful when comparing weak-form implementations.

## Project Structure

```text
.
├── 1D_Allen-Cahn_EvoKAN-SF.ipynb
├── 1D_Allen-Cahn_EvoKAN_WF.ipynb
├── 2D_Heat_EvoKAN_SF.ipynb
├── 2D_Heat_EvoKAN_WF.ipynb
├── 2D_Heat_EvoKAN_WF_2.ipynb
├── Weak_form_evolutionary_KAN (6).pdf
├── requirements.txt
├── .gitignore
├── results/
│   └── ...
├── trained_model/
│   └── ...
└── scripts/
```

## Citation

If you use this repository, please cite the associated weak-form evolutionary KAN paper and reference this repository.

At minimum, include:

- the paper distributed in this repository: `Weak_form_evolutionary_KAN (6).pdf`
- the repository name: `EvoKAN-WF`

## License

Please use the same license that is configured for the GitHub repository.

