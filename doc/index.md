# Sim4Life FEniCS Plugin Documentation

## Introduction

The FEniCS solver in this plugin provides a modular finite element simulation backend for Sim4Life. It leverages the FEniCS finite element library to solve a wide range of partial differential equations (PDEs) defined within the Sim4Life environment. The solver supports stationary (steady-state), eigenvalue, and time-domain (transient) simulations, making it suitable for a variety of physical modeling tasks.

### Key Features

- **General PDE Solving**: Supports the solution of general PDEs using the finite element method.
- **Multiple Simulation Types**: Handles stationary, eigenvalue, and time-domain problems.
- **Sim4Life Integration**: Reads simulation definitions, meshes, and boundary conditions exported from Sim4Life.
- **Automated Workflow**: Imports mesh and model data, assembles variational forms, applies boundary conditions, solves the problem, and exports results for visualization.
- **Extensible Design**: Modular structure allows for easy extension with new equations, boundary conditions, and post-processing routines.
- **Real and Complex Problems**: Supports both real-valued and complex-valued simulations, enabling a wide range of physical applications.

### Workflow Overview

1. **Input Preparation**: Simulation definitions and mesh data are exported from Sim4Life as JSON and VTU files.
2. **Solver Initialization**: The main solver script loads the input, prepares the mesh, and sets up the problem.
3. **Problem Assembly**: Function spaces, variational forms, and boundary conditions are created based on the user-defined model.
4. **Solving**: The appropriate solver is selected based on the simulation type (stationary, eigenvalue, or time-domain).
5. **Result Export**: Simulation results are written to VTK files for visualization and further analysis.

This design enables users to run complex finite element simulations within Sim4Life, while benefiting from the flexibility and power of the FEniCS library.

### Seamless Postprocessing Integration

Simulation results produced by the FEniCS solver are fully compatible with the Sim4Life postprocessing pipeline. Results can be directly viewed, inspected, and analyzed using the integrated Sim4Life viewers, allowing for efficient visualization and further analysis without additional conversion steps.

---

## Supported Equation Categories

The Sim4Life FEniCS plugin supports several categories of equations, each tailored to common classes of physics problems:

### 1. **PDE (Partial Differential Equation)**
- **Purpose:** Solve standard scalar PDEs such as diffusion, heat conduction, and Poisson’s equation.
- **User Inputs:** Diffusion coefficient, linear term, source term, divergence term, boundary conditions.
- **Example Equation:**  
  $$
  \nabla \cdot (c \nabla u) + l u + \nabla \cdot \mathbf{M} + f = 0
  $$
- **Typical Applications:**  
  - Heat transfer: $c$ = thermal conductivity, $u$ = temperature  
  - Electrostatics: $c$ = permittivity, $u$ = potential  
  - Diffusion: $c$ = diffusion coefficient, $u$ = concentration

### 2. **Solid Mechanics**
- **Purpose:** Model linear elasticity and structural mechanics problems.
- **User Inputs:** Young’s modulus, Poisson’s ratio, body force, thermal expansion, displacement/traction boundary conditions.
- **Example Equation:**  
  $$
  \nabla \cdot \sigma + \mathbf{F} = 0
  $$
  where $\sigma$ is the stress tensor, $\mathbf{F}$ is the body force.
- **Typical Applications:**  
  - Deformation of solids under load  
  - Thermoelasticity  
  - Biomechanics

### 3. **Weak Form**
- **Purpose:** Directly specify custom variational (weak) forms for advanced or non-standard problems.
- **User Inputs:** UFL (Unified Form Language) expressions for bilinear and linear forms, integration domains, boundary conditions.
- **Example Equation:**  
  $$
  \int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx
  $$
  (or any custom weak form)
- **Typical Applications:**  
  - Stokes flow (fluid dynamics)  
  - Mixed finite element methods  
  - Custom multiphysics couplings

---

## Example Applications

| Example                | Category      | Physics/Equation Type                |
|------------------------|--------------|--------------------------------------|
| Harmonic Oscillator    | PDE          | Quantum mechanics, eigenvalue problem|
| Magnetic Dipole        | PDE/Weak Form| Magnetostatics, Poisson’s equation   |
| Quantum Tunneling      | PDE          | Time-dependent Schrödinger equation  |
| Stokes Flow            | Weak Form    | Fluid dynamics, Stokes equations     |
| Deformed Cantilever    | Solid Mechanics | Thermoelasticity, linear elasticity   |

---

## How It Works

1. **Create a Simulation:** Select "FEniCS" as the simulation type in Sim4Life.
2. **Add Equations:** Choose from PDE, Solid Mechanics, or Weak Form. Assign them to mesh domains.
3. **Set Parameters:** Enter coefficients, material properties, or custom expressions as needed.
4. **Define Boundary Conditions:** Apply Dirichlet (fixed value) or Neumann/flux (derivative) conditions to boundaries.
5. **Run and Analyze:** Solve the problem and visualize results within Sim4Life.

---

## Examples

- [Harmonic Oscillator: Quantum Eigenvalues](examples/harmonic_oscillator/README.md)
- [Magnetic Dipole: Magnetized Sphere](examples/magnetic_dipole/README.md)
- [Quantum Tunneling](examples/quantum_tunneling/README.md)
- [Stokes Flow Past a Sphere](examples/stokes_flow/README.md)
- [Deformed Cantilever: Thermoelastic Beam](examples/deformed_cantilever/README.md)

Explore each example for ready-to-use simulation setups and meshes.
