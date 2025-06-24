# FEniCS Simulation Plugin for Sim4Life

## Overview

This plugin integrates the FEniCS finite element library into Sim4Life, enabling advanced multiphysics simulations directly within the Sim4Life environment. It provides a flexible and powerful framework for modeling a wide range of physical phenomena using the finite element method (FEM).

With this plugin, users can set up, solve, and analyze partial differential equations (PDEs) and custom variational problems, leveraging FEniCS’s capabilities through an intuitive graphical interface.

---

## Supported Equation Categories

The plugin supports several categories of equations, each tailored to common classes of physics problems:

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

---

## How It Works

1. **Create a Simulation:** Select "FEniCS" as the simulation type in Sim4Life.
2. **Add Equations:** Choose from PDE, Solid Mechanics, or Weak Form. Assign them to mesh domains.
3. **Set Parameters:** Enter coefficients, material properties, or custom expressions as needed.
4. **Define Boundary Conditions:** Apply Dirichlet (fixed value) or Neumann/flux (derivative) conditions to boundaries.
5. **Run and Analyze:** Solve the problem and visualize results within Sim4Life.

---

## Technical Implementation

- Modular model/controller/solver architecture
- FEniCS backend for FEM assembly and solution
- Seamless integration with Sim4Life’s geometry and UI
- Support for custom expressions and advanced boundary conditions

---

## Requirements

- Sim4Life core application
- Python 3.11 or higher
- FEniCS library (compatible version)

---

For detailed example setups, see the `doc/examples` folder. Each example includes a physical background, model setup, and instructions for loading into Sim4Life.

