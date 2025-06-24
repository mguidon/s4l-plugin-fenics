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

## Examples

- [Example 1: Poisson Equation on a Unit Square](examples/example1/README.md)
- [Example 2: Linear Elasticity in 2D](examples/example2/README.md)
- [Example 3: Heat Equation (Transient)](examples/example3/README.md)
- [Example 4: Laplace Equation on a Circle](examples/example4/README.md)

Explore each example for ready-to-use simulation setups and meshes.
