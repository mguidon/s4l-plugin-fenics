import dataclasses
import json
import logging
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import api_models as mdl
import dolfinx
import meshio
import numpy
import pyvista
import ufl
from dolfinx import fem, io, plot
from dolfinx.fem import petsc
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io.utils import XDMFFile
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.plotting.plotter import Plotter as pv_Plotter
from slepc4py import SLEPc
from ufl.variable import Variable

TETRA_EL_NAME = "tetra"
TRIANGLE_EL_NAME = "triangle"

LOGGER = logging.getLogger("main")


def get_logger() -> logging.Logger:
    return LOGGER


@dataclass
class MeshData:
    mesh: dolfinx.mesh.Mesh
    domain_tags: dolfinx.mesh.MeshTags
    patch_tags: dolfinx.mesh.MeshTags
    surface_mesh: dolfinx.mesh.Mesh
    patch_tags_on_surface_mesh: dolfinx.mesh.MeshTags

    def display_mesh(
        self, block: bool = True, save_path: Path = Path("displayed_mesh.pdf")
    ) -> None:
        pyvista.start_xvfb()  # type: ignore

        V = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 1))  # type: ignore
        cells, types, x = plot.vtk_mesh(V)
        domain_tags_grid = pyvista.UnstructuredGrid(cells, types, x)
        domain_tags_grid.cell_data["Domain Marker"] = self.domain_tags.values
        domain_tags_grid.set_active_scalars("Domain Marker")

        V_surface = dolfinx.fem.functionspace(self.surface_mesh, ("Lagrange", 1))  # type: ignore
        cells, types, x = plot.vtk_mesh(V_surface)
        patch_tags_grid = pyvista.UnstructuredGrid(cells, types, x)
        patch_tags_grid.cell_data[
            "Patch Marker"
        ] = self.patch_tags_on_surface_mesh.values
        patch_tags_grid.set_active_scalars("Patch Marker")

        p = pv_Plotter(shape=(1, 2))
        p.subplot(0, 0)
        p.add_mesh_clip_plane(domain_tags_grid, show_edges=False, show_scalar_bar=True)
        p.show_axes()  # type: ignore
        p.show_grid()  # type: ignore

        p.subplot(0, 1)
        p.add_mesh_clip_plane(patch_tags_grid, show_edges=False, show_scalar_bar=True)
        p.show_axes()  # type: ignore
        p.show_grid()  # type: ignore

        if block:
            print("blocking on plot window ... close it to continue")
            p.show()
        else:
            p.save_graphic(save_path)


ufl_eq_t = Any  # TODO
variables_t = dict[str, ufl.Argument | ufl.Coargument]


@dataclass
class FenicsxProblem:
    results_dir: Path | None = None
    mesh_data: MeshData | None = None
    function_space: fem.FunctionSpaceBase | None = None
    element: ufl.FiniteElementBase | None = None
    dirichlet_conditions: list[fem.DirichletBC] | None = None
    dx: ufl.Measure | None = None
    ds: ufl.Measure | None = None
    x: ufl.SpatialCoordinate | None = None
    u: fem.Function | None = None  # stationary problem result
    EPS: SLEPc.EPS | None = None  # eigenvalue problem result  # type: ignore
    variables: variables_t | None = None
    test_variables: variables_t | None = None
    a: ufl_eq_t | None = None
    L: ufl_eq_t | None = None
    time: Variable | None = None
    time_step: Variable | None = None


# Functions


def _create_mesh(
    filepath: Path,
    cell_type_name: str,
    tag_data_name: str,
) -> meshio.Mesh:
    in_mesh: meshio.Mesh = meshio.read(str(filepath))
    cells = in_mesh.get_cells_type(cell_type_name)
    cell_data = in_mesh.get_cell_data(tag_data_name, cell_type_name)
    points = in_mesh.points
    return meshio.Mesh(
        points=points,
        cells={cell_type_name: cells},
        cell_data={"name_to_read": [cell_data]},
    )


def info_log(msg: str, rank0_only: bool = False):
    if rank0_only and MPI.COMM_WORLD.rank != 0:
        return

    LOGGER.info(f"[{MPI.COMM_WORLD.rank}] {msg}")


class TimingContext:
    def __init__(self, desc: str, indent=0) -> None:
        self._desc = desc
        self._T0 = 0
        self._indent = indent

    def __enter__(self):
        self._T0 = perf_counter()
        info_log(f"{'--'*self._indent}--> {self._desc} started")
        return self

    def __exit__(self, *args, **kwargs):
        info_log(
            f"<{'--'*self._indent}-- {self._desc} finished in {perf_counter() - self._T0:.1f}s"
        )


def import_s4l_mesh(
    mesh_domains: Path, mesh_patches: Path, results_dir: Path
) -> MeshData:
    rank = MPI.COMM_WORLD.rank
    num_proc = MPI.COMM_WORLD.size

    info_log(f"Importing mesh across {num_proc} processes", rank0_only=True)

    domain_mesh_path = results_dir / "domain_mesh.xdmf"
    patch_mesh_path = results_dir / "patch_mesh.xdmf"

    if rank == 0:
        with TimingContext("prepare XDMF", 1):
            with TimingContext("domains", 2):
                domain_mesh = _create_mesh(mesh_domains, TETRA_EL_NAME, "Domains")
                meshio.write(domain_mesh_path, domain_mesh)

            with TimingContext("patches", 2):
                patch_mesh = _create_mesh(mesh_patches, TRIANGLE_EL_NAME, "Patches")
                meshio.write(patch_mesh_path, patch_mesh)

    MPI.COMM_WORLD.barrier()

    with TimingContext("read XDMF", 1):
        with TimingContext("domains", 2):
            with XDMFFile(MPI.COMM_WORLD, domain_mesh_path, "r") as xdmf:
                with TimingContext("mesh", 3):
                    mesh = xdmf.read_mesh(name="Grid")
                with TimingContext("meshtags", 3):
                    domain_tags = xdmf.read_meshtags(mesh, name="Grid")

        with TimingContext("patches", 2):
            with TimingContext("initialise connectivity", 2):
                mesh.topology.create_connectivity(
                    mesh.topology.dim, mesh.topology.dim - 1
                )

            with XDMFFile(MPI.COMM_WORLD, patch_mesh_path, "r") as xdmf:
                with TimingContext("meshtags", 3):
                    patch_tags = xdmf.read_meshtags(mesh, name="Grid")

        # load surface for later visualisation

        with TimingContext("patches - visualisation", 2):
            with XDMFFile(MPI.COMM_WORLD, patch_mesh_path, "r") as xdmf:
                surface_mesh = xdmf.read_mesh(name="Grid")
                patch_tags_on_surface_mesh = xdmf.read_meshtags(
                    surface_mesh, name="Grid"
                )

    return MeshData(
        mesh=mesh,
        domain_tags=domain_tags,
        patch_tags=patch_tags,
        surface_mesh=surface_mesh,
        patch_tags_on_surface_mesh=patch_tags_on_surface_mesh,
    )


#


def element_from_model(
    mesh: dolfinx.mesh.Mesh, eq: mdl.Equation
) -> ufl.FiniteElementBase:
    if eq.variable_shape == mdl.VariableType.SCALAR.value:
        element = ufl.FiniteElement(eq.element_type, mesh.ufl_cell(), eq.element_degree)
    elif eq.variable_shape == mdl.VariableType.VECTOR.value:
        element = ufl.VectorElement(eq.element_type, mesh.ufl_cell(), eq.element_degree)
    else:
        raise ValueError(f"Unexpected variable_shape: {eq.variable_shape}")

    return element


def mixed_element_from_model(
    mesh: dolfinx.mesh.Mesh, eqs: list[mdl.Equation]
) -> ufl.FiniteElementBase:
    elements = []
    for eq in eqs:
        elements.append(element_from_model(mesh, eq))

    return ufl.MixedElement(elements)


def function_space_from_model(
    mesh: dolfinx.mesh.Mesh, simulation_mdl: mdl.Simulation
) -> fem.FunctionSpaceBase:
    mixed_element = mixed_element_from_model(mesh, simulation_mdl.equations)
    func_space = fem.FunctionSpace(mesh, mixed_element)

    return func_space


def dirichlet_conditions_from_model(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> list[fem.DirichletBC]:
    bcs: list[fem.DirichletBC] = []

    for equation_index, equation in enumerate(simulation_mdl.equations):
        info_log(
            f"Defining dirichlet conditions for: {equation.variable_name}",
            rank0_only=True,
        )
        for bc_mdl in equation.dirichlet_conditions:
            bcs += boundary_condition_from_model(
                problem, equation_index, bc_mdl, simulation_mdl.global_expressions
            )

    return bcs


def init_problem_variables(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> None:
    assert problem.function_space is not None
    assert problem.mesh_data is not None

    problem.u = fem.Function(problem.function_space)  # type: ignore

    info_log(f"Problem has {len(problem.u.x.array)} DOF's (before applying bc's) on this rank")  # type: ignore

    #

    problem.variables = {}

    if simulation_mdl.solver_settings.solver_type == mdl.SolverType.LINEAR.value:
        for eq_idx, sub_trial_func in enumerate(ufl.TrialFunctions(problem.function_space)):  # type: ignore
            problem.variables[simulation_mdl.equations[eq_idx].variable_name] = sub_trial_func  # type: ignore

    elif simulation_mdl.solver_settings.solver_type == mdl.SolverType.NON_LINEAR.value:
        for eq_idx, sub_func in enumerate(ufl.split(problem.u)):
            problem.variables[simulation_mdl.equations[eq_idx].variable_name] = sub_func  # type: ignore

    else:
        raise ValueError(
            f"Unhandled solver type: {simulation_mdl.solver_settings.solver_type}"
        )

    #

    problem.test_variables = {}
    for eq_idx, sub_trial_func in enumerate(ufl.TestFunctions(problem.function_space)):  # type: ignore
        problem.test_variables[simulation_mdl.equations[eq_idx].variable_name + "_test"] = sub_trial_func  # type: ignore

    #

    problem.x = ufl.SpatialCoordinate(problem.mesh_data.mesh)
    problem.dx = ufl.Measure(
        "dx", problem.mesh_data.mesh, subdomain_data=problem.mesh_data.domain_tags
    )
    problem.ds = ufl.Measure(
        "ds", domain=problem.mesh_data.mesh, subdomain_data=problem.mesh_data.patch_tags
    )


context_t = Optional[dict[str, Any]]


def add_expressions_to_context(
    context: dict[str, Any], expressions: dict[str, str]
) -> None:
    for expr in expressions:
        if expr in context:
            raise RuntimeError(f"Existing value found for global expression: {expr}")

        context[expr] = eval(expressions[expr], globals(), context)


def a_L_for_weak_term_from_model(
    problem: FenicsxProblem,
    global_expressions: dict[str, str],
    weak_term: mdl.WeakTerm,
    context: context_t = None,
) -> tuple[ufl_eq_t, ufl_eq_t]:
    assert problem.variables is not None and problem.test_variables is not None
    assert problem.mesh_data is not None

    if context is None:
        context = dict(
            x=problem.x,
            dx=problem.dx,
            ds=problem.ds,
            _mesh_=problem.mesh_data.mesh,
            pi=math.pi,
        )
    else:
        context = context.copy()

    for variable in problem.variables:
        assert variable not in context, f"Duplicate: {variable}"
        context[variable] = problem.variables[variable]

    for test_variable in problem.test_variables:
        assert test_variable not in context, f"Duplicate: {test_variable}"
        context[test_variable] = problem.test_variables[test_variable]

    add_expressions_to_context(context, global_expressions)

    a = 0
    for idx, term in enumerate(weak_term.a_terms):
        info_log(f"-- defining bilinear (a) term: {idx}: {term}", rank0_only=True)
        a += eval(term, globals(), context)

    L = 0
    for idx, term in enumerate(weak_term.L_terms):
        info_log(f"-- defining linear (L) term: {idx}: {term}", rank0_only=True)
        L += eval(term, globals(), context)

    return a, L


def a_L_from_equation(
    problem: FenicsxProblem,
    equation_mdl: mdl.Equation,
    global_expressions: dict[str, str],
    context: context_t = None,
) -> tuple[ufl_eq_t, ufl_eq_t]:
    a = 0
    L = 0
    for idx, weak_term in enumerate(equation_mdl.weak_form):
        info_log(f"- Defining weak_term {idx}", rank0_only=True)

        a_term, L_term = a_L_for_weak_term_from_model(
            problem, global_expressions, weak_term, context=context
        )
        a += a_term
        L += L_term

    return a, L


def a_L_from_equations(
    problem: FenicsxProblem,
    equations: list[mdl.Equation],
    global_expressions: dict[str, str],
    context: context_t = None,
) -> tuple[ufl_eq_t, ufl_eq_t]:
    a = 0
    L = 0
    for equ_idx, equation in enumerate(equations):
        info_log(
            f"Defining equation {equ_idx}: {equation.variable_name}", rank0_only=True
        )
        a_eq, L_eq = a_L_from_equation(problem, equation, global_expressions, context)

        a += a_eq
        L += L_eq

    return a, L


def a_L_from_model(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> tuple[ufl_eq_t, ufl_eq_t]:
    return a_L_from_equations(
        problem, simulation_mdl.equations, simulation_mdl.global_expressions
    )


def boundary_condition_from_model(
    problem: FenicsxProblem,
    equation_index: int,
    bc_mdl: mdl.DirichletCondition,
    global_expressions: dict[str, str],
) -> list[fem.DirichletBC]:
    assert problem.function_space is not None and problem.mesh_data is not None
    assert problem.variables is not None

    context: dict[str, Any] = dict(x=problem.x)

    for variable in problem.variables:
        assert variable not in context, f"Duplicate: {variable}"
        context[variable] = problem.variables[variable]

    for expr in global_expressions:
        if expr in context:
            raise RuntimeError(f"Existing value found for global expression: {expr}")

        context[expr] = eval(global_expressions[expr], globals(), context)

    info_log(
        f"Defining bc: {bc_mdl.value} on patches: {bc_mdl.domain_indices}",
        rank0_only=True,
    )

    sub_space = problem.function_space.sub(equation_index)
    collapsed_sub_space, _ = sub_space.collapse()

    form = eval(bc_mdl.value, globals(), context)

    # n.b. we ensure that we "upgrade" the user supplied string to a form by adding Constant(0) to it
    if bc_mdl.value_type == mdl.VariableType.SCALAR.value:
        form += fem.Constant(problem.mesh_data.mesh, dolfinx.default_scalar_type(0))
    elif bc_mdl.value_type == mdl.VariableType.VECTOR.value:
        form += fem.Constant(
            problem.mesh_data.mesh, dolfinx.default_scalar_type((0, 0, 0))
        )
    else:
        raise ValueError(f"Unexpected bc value_type: {bc_mdl.value_type}")

    expr = fem.Expression(form, collapsed_sub_space.element.interpolation_points())
    func = fem.Function(collapsed_sub_space)
    func.interpolate(expr)  # type: ignore

    bcs = []

    for domain_index in bc_mdl.domain_indices:
        dofs = fem.locate_dofs_topological(
            (sub_space, collapsed_sub_space),
            # sub_space,
            problem.mesh_data.mesh.topology.dim - 1,
            problem.mesh_data.patch_tags.find(domain_index),
        )

        bcs.append(
            fem.dirichletbc(
                func,  # type: ignore
                dofs,
                sub_space,
            )
        )

    return bcs


def solve_problem(problem: FenicsxProblem, simulation_mdl: mdl.Simulation) -> bool:
    if simulation_mdl.simulation_type == mdl.SimulationType.STATIONARY.value:
        return solve_stationary_problem(problem, simulation_mdl.solver_settings)
    elif simulation_mdl.simulation_type == mdl.SimulationType.EIGENVALUE.value:
        return solve_eigenvalue_problem(problem, simulation_mdl)
    else:
        raise ValueError(
            f"Unexpected simulation type: {simulation_mdl.simulation_type}"
        )


def configure_ksp_options_dict(
    solver_settings: mdl.SolverSettings,
    verbose: bool,
    options_prefix: str = "",
    options_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert (
        solver_settings.linear_solver_type == mdl.LinearSolverType.ITERATIVE_KSP.value
    )

    if options_dict is None:
        options_dict = {}

    options_dict[f"{options_prefix}ksp_type"] = solver_settings.ksp_type
    options_dict[f"{options_prefix}ksp_rtol"] = solver_settings.ksp_rtol
    options_dict[f"{options_prefix}ksp_atol"] = solver_settings.ksp_atol

    if solver_settings.ksp_pc == mdl.KSPPreConditionerType.EUCLID.value:
        options_dict[f"{options_prefix}pc_type"] = "hypre"
        options_dict[f"{options_prefix}pc_hypre_type"] = "euclid"
    elif solver_settings.ksp_pc == mdl.KSPPreConditionerType.PILUT.value:
        options_dict[f"{options_prefix}pc_type"] = "hypre"
        options_dict[f"{options_prefix}pc_hypre_type"] = "pilut"
    elif solver_settings.ksp_pc == mdl.KSPPreConditionerType.BOOMER_AMG.value:
        options_dict[f"{options_prefix}pc_type"] = "hypre"
        options_dict[f"{options_prefix}pc_hypre_type"] = "boomeramg"
    elif solver_settings.ksp_pc in (
        mdl.KSPPreConditionerType.NONE.value,
        mdl.KSPPreConditionerType.ILU.value,
    ):
        options_dict[f"{options_prefix}pc_type"] = solver_settings.ksp_pc
    else:
        raise ValueError(f"Unhandled pre-conditioner type: {solver_settings.ksp_pc}")

    if verbose:
        options_dict[f"{options_prefix}ksp_monitor_short"] = ""

    return options_dict


def configure_petsc_for_lu(ksp) -> None:
    # TODO pass these as petsc options?
    ksp.setType("preonly")

    # Configure MUMPS to handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()

    # https://petsc.org/release/manualpages/Mat/MATSOLVERMUMPS/
    # -mat_mumps_icntl_24 - ICNTL(24): detection of null pivot rows (0 or 1)
    # -mat_mumps_icntl_25 - ICNTL(25): compute a solution of a deficient matrix and a null space basis
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)


def solve_stationary_problem(
    problem: FenicsxProblem, solver_settings: mdl.SolverSettings, verbose: bool = True
) -> bool:
    assert problem.u is not None
    assert problem.a is not None
    assert problem.L is not None
    assert problem.dirichlet_conditions is not None
    assert problem.results_dir is not None

    solver_type = solver_settings.solver_type
    linear_solver_type = solver_settings.linear_solver_type

    petsc_log = problem.results_dir / "petsc.log"

    if verbose:
        info_log(f"Solving problem with {solver_type} solver")

    T0 = perf_counter()

    if solver_type == mdl.SolverType.LINEAR.value:

        linear_solver_configured = False
        petsc_options = {}
        if linear_solver_type == mdl.LinearSolverType.ITERATIVE_KSP.value:
            petsc_options = configure_ksp_options_dict(solver_settings, verbose)
            linear_solver_configured = True

        linear_problem = petsc.LinearProblem(
            problem.a,
            problem.L,
            u=problem.u,
            bcs=problem.dirichlet_conditions,
            petsc_options=petsc_options,
        )

        if linear_solver_type == mdl.LinearSolverType.DIRECT_LU.value:
            configure_petsc_for_lu(linear_problem.solver)
            linear_solver_configured = True

        if not linear_solver_configured:
            raise ValueError(f"Unexpected linear solver type: {linear_solver_type}")

        problem.u = linear_problem.solve()

        if verbose:
            viewer = PETSc.Viewer().createASCII(str(petsc_log))  # type: ignore
            linear_problem.solver.view(viewer)

            info_log(f"Linear solver finished in {perf_counter()-T0:.1f}s", True)

        converged = True
    elif solver_type == mdl.SolverType.NON_LINEAR.value:
        petsc_problem = petsc.NonlinearProblem(
            problem.a - problem.L, u=problem.u, bcs=problem.dirichlet_conditions
        )
        solver = NewtonSolver(MPI.COMM_WORLD, petsc_problem)

        solver.rtol = solver_settings.nonlin_rtol
        solver.atol = solver_settings.nonlin_atol

        ksp = solver.krylov_solver
        petsc_options = PETSc.Options()  # type: ignore
        option_prefix = ksp.getOptionsPrefix()

        if linear_solver_type == mdl.LinearSolverType.ITERATIVE_KSP.value:
            configure_ksp_options_dict(
                solver_settings,
                verbose,
                options_dict=petsc_options,
                options_prefix=option_prefix,
            )
        elif linear_solver_type == mdl.LinearSolverType.DIRECT_LU.value:
            pass  # this is the default
        else:
            raise ValueError(f"Unexpected linear solver type: {linear_solver_type}")

        ksp.setFromOptions()

        iters, converged = solver.solve(problem.u)
        viewer = PETSc.Viewer().createASCII(str(petsc_log))  # type: ignore
        solver.krylov_solver.view(viewer)

        if converged and verbose:
            info_log(
                f"Non-linear solver converged in {iters} iterations (and {perf_counter()-T0:.1f}s)",
                rank0_only=True,
            )
        elif not converged:
            LOGGER.error("Non-linear solver failed to converge")

    else:
        raise ValueError(f"Unhandled solver type: {solver_type}")

    if verbose:
        if petsc_log.is_file():
            info_log(petsc_log.read_text())
        else:
            LOGGER.error("Failed to find petsc log")
    MPI.COMM_WORLD.barrier()

    return converged


def solve_eigenvalue_problem(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> bool:
    assert problem.u is not None
    assert problem.a is not None
    assert problem.L is not None
    assert problem.dirichlet_conditions is not None
    assert problem.results_dir is not None

    info_log(
        "Solving eigenvalue problem with KRYLOV-SCHUR solver"
    )  # TODO: Add other solver types (EPS.setType) to settings
    T0 = perf_counter()

    A = assemble_matrix(fem.form(problem.a), bcs=problem.dirichlet_conditions)
    A.assemble()

    B = assemble_matrix(
        fem.form(problem.L),
        bcs=problem.dirichlet_conditions,
        diagonal=0.0,  # avoid spurious lambda = 1 modes
    )
    B.assemble()

    EPS = SLEPc.EPS()  # type: ignore
    EPS.create(comm=MPI.COMM_WORLD)
    EPS.setOperators(A, B)
    EPS.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # type: ignore
    EPS.setType(SLEPc.EPS.Type.KRYLOVSCHUR)  # type: ignore
    EPS.setDimensions(nev=simulation_mdl.solver_settings.num_eigenvals)
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # type: ignore
    EPS.setTarget(simulation_mdl.solver_settings.target_eigenval)
    EPS.setTolerances(tol=simulation_mdl.solver_settings.eigenval_tol, max_it=50)

    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)  # type: ignore
    ST.setShift(simulation_mdl.solver_settings.target_eigenval)
    EPS.setST(ST)
    EPS.view()

    EPS.setMonitor(
        lambda eps, it, nconv, eig, err: info_log(
            f"Eigenvalue solver: it={it}, nconv: {nconv} of {simulation_mdl.solver_settings.num_eigenvals}",
            True,
        )
    )
    EPS.setFromOptions()

    T0 = perf_counter()
    EPS.solve()
    info_log(f"Eigenvalue solver finished in {perf_counter()-T0:.1f}s", True)

    problem.EPS = EPS

    return simulation_mdl.solver_settings.num_eigenvals <= EPS.getConverged()


def export_solved_problem(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> None:
    if simulation_mdl.simulation_type == mdl.SimulationType.STATIONARY.value:
        return export_solved_stationary_problem(problem, simulation_mdl)
    elif simulation_mdl.simulation_type == mdl.SimulationType.EIGENVALUE.value:
        return export_solved_eigenvalue_problem(problem, simulation_mdl)
    else:
        raise ValueError(
            f"Unexpected simulation type: {simulation_mdl.simulation_type}"
        )


def export_solved_stationary_problem(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> None:
    assert problem.results_dir is not None
    assert problem.mesh_data is not None
    assert problem.u is not None

    for eq_idx, equation in enumerate(simulation_mdl.equations):
        with io.VTKFile(  # pyright: ignore[reportPrivateImportUsage]
            MPI.COMM_WORLD, problem.results_dir / f"{equation.variable_name}.vtu", "w"
        ) as file:
            file.write_function(problem.u.sub(eq_idx).collapse())


def export_solved_eigenvalue_problem(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> None:
    assert problem.results_dir is not None
    assert problem.mesh_data is not None
    assert problem.EPS is not None
    assert problem.function_space is not None

    # Collect
    num_eigvals = problem.EPS.getConverged()

    eigvals = [problem.EPS.getEigenvalue(i) for i in range(num_eigvals)]

    eigvecs_real, eigvecs_imag = [], []
    errors = []

    assert len(simulation_mdl.equations) == 1, "TODO"
    V0 = problem.function_space.sub(0).collapse()[0]

    for index in range(num_eigvals):
        vr = fem.Function(V0)
        vi = fem.Function(V0)

        problem.EPS.getEigenvector(index, vr.vector, vi.vector)  # type: ignore

        eigvecs_real.append(vr)
        eigvecs_imag.append(vi)
        errors.append(problem.EPS.computeError(index))

    # Sort
    sorted_indices = numpy.argsort(numpy.real(numpy.array(eigvals)), axis=0)
    eigvals = [eigvals[index] for index in sorted_indices]
    eigvecs_real = [eigvecs_real[index] for index in sorted_indices]
    eigvecs_imag = [eigvecs_imag[index] for index in sorted_indices]
    errors = [errors[index] for index in sorted_indices]

    # Export

    metadata = []
    for index in range(num_eigvals):
        metadata.append(
            mdl.EigenvectorMetadata(
                eigenvalue_real=numpy.real(eigvals[index]),
                eigenvalue_imag=numpy.imag(eigvals[index]),
                error=errors[index],
            )
        )

    all_metadata = mdl.EigenvectorsMetadata(metadata=metadata)

    with open(problem.results_dir / "eigenvectors.json", "w") as fh:
        json.dump(dataclasses.asdict(all_metadata), fh, indent=4)

    for index in range(num_eigvals):

        with io.VTKFile(  # pyright: ignore[reportPrivateImportUsage]
            MPI.COMM_WORLD, problem.results_dir / f"eig{index}_real.vtu", "w"
        ) as file:
            file.write_function(eigvecs_real[index])  # type: ignore

        with io.VTKFile(  # pyright: ignore[reportPrivateImportUsage]
            MPI.COMM_WORLD, problem.results_dir / f"eig{index}_imag.vtu", "w"
        ) as file:
            file.write_function(eigvecs_imag[index])  # type: ignore


def export_postpro_exprs(
    problem: FenicsxProblem, simulation_mdl: mdl.Simulation
) -> None:

    if simulation_mdl.simulation_type == mdl.SimulationType.STATIONARY.value:
        return export_stationary_postpro_exprs(
            simulation_mdl.global_expressions,
            problem,
            simulation_mdl.post_processing_expressions,
        )
    elif simulation_mdl.simulation_type == mdl.SimulationType.EIGENVALUE.value:
        return export_eigenvalue_postpro_exprs(
            simulation_mdl.global_expressions,
            problem,
            simulation_mdl.post_processing_expressions,
        )
    else:
        raise ValueError(
            f"Unexpected simulation type: {simulation_mdl.simulation_type}"
        )


def export_postpro_exprs_impl(
    postpro_expressions: dict[str, str],
    context: dict[str, Any],
    mesh: dolfinx.mesh.Mesh,
    vtk_files: list[io.VTKFile],  # pyright: ignore[reportPrivateImportUsage]
    time: float | None = None,
) -> None:

    for pp_idx, pp_variable in enumerate(postpro_expressions):
        form = eval(postpro_expressions[pp_variable], globals(), context)
        if form.ufl_shape == ():
            func_space = fem.functionspace(mesh, ("CG", 1))  # type: ignore
            form += fem.Constant(mesh, dolfinx.default_scalar_type(0))
        elif form.ufl_shape == (3,):
            vec_element = ufl.VectorElement("CG", mesh.ufl_cell(), 1)
            func_space = fem.FunctionSpace(mesh, vec_element)
            form += fem.Constant(mesh, dolfinx.default_scalar_type((0, 0, 0)))
        else:
            raise ValueError(
                "Unsupported dimensionality of evaluated expression (must be scalar or vector)"
            )

        if time is not None:
            form = ufl.replace(form, {context["time"]: time})

        expr = fem.Expression(form, func_space.element.interpolation_points())
        func = fem.Function(func_space)
        func.interpolate(expr)  # type: ignore

        vtk_files[pp_idx].write_function(func, 0 if time is None else time)  # type: ignore


def export_stationary_postpro_exprs(
    global_expressions: dict[str, str],
    problem: FenicsxProblem,
    postpro_expressions: dict[str, str],
) -> None:
    assert problem.results_dir is not None
    assert problem.u is not None
    assert problem.variables is not None
    assert problem.mesh_data is not None

    context: dict[str, Any] = dict(x=problem.x)
    sub_funcs = ufl.split(problem.u)
    for index, variable in enumerate(problem.variables):
        context[variable] = sub_funcs[index]

    add_expressions_to_context(context, global_expressions)

    with ExitStack() as stack:

        vtk_files = [
            stack.enter_context(
                io.VTKFile(  # pyright: ignore[reportPrivateImportUsage]
                    MPI.COMM_WORLD, problem.results_dir / f"{pp_variable}.vtu", "w"
                )
            )
            for pp_variable in postpro_expressions
        ]

        export_postpro_exprs_impl(
            postpro_expressions,
            context,
            problem.mesh_data.mesh,
            vtk_files,
        )


def export_eigenvalue_postpro_exprs(
    global_expressions: dict[str, str],
    problem: FenicsxProblem,
    postpro_expressions: dict[str, str],
) -> None:
    assert problem.results_dir is not None
    assert problem.variables is not None
    assert problem.mesh_data is not None
    assert problem.mesh_data.mesh is not None

    info_log(
        "PostPro expressions not supported for eigenvalue problems ... skipping"
    )  # TODO: set u to each eigenvalue and then evaluate expression and export
