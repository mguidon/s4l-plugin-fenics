import dataclasses
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import api_models as mdl
import irksome.ButcherTableaux as bt
import numpy
import ufl
from core_utils import TqdmLogger
from dolfinx import default_scalar_type, fem, io
from dolfinx.mesh import Mesh
from mpi4py import MPI
from simulation_utils import (
    FenicsxProblem,
    MeshData,
    a_L_from_equations,
    add_expressions_to_context,
    export_postpro_exprs_impl,
    get_logger,
    info_log,
    mixed_element_from_model,
    solve_stationary_problem,
    variables_t,
)
from tqdm import tqdm as Tqdm
from ufl.algorithms.ad import expand_derivatives
from ufl.classes import Zero as ufl_Zero  # type: ignore
from ufl.core.expr import Expr
from ufl.core.expr import Expr as ufl_Expr

SMOOTHING_FACTOR = (
    1 / 10  # =(1/time constant) for exponential smoothing (Proportional control)
)

context_t = dict[str, Any]
expressions_t = dict[str, str]


def to_scalar_form(expr: str, context: context_t) -> Expr:
    form = eval(expr, globals(), context)
    form += fem.Constant(context["_mesh_"], default_scalar_type(0))
    return form


def to_vector_form(expr: str, context: context_t) -> Expr:
    form = eval(expr, globals(), context)
    form += fem.Constant(context["_mesh_"], default_scalar_type((0, 0, 0)))
    return form


def to_function(form, V) -> fem.Function:
    expr = fem.Expression(form, V.element.interpolation_points())
    func = fem.Function(V)
    assert isinstance(func, fem.Function)
    func.interpolate(expr)
    return func


def export_time_step(
    timesteps_metadata: mdl.TimestepsMetadata,
    u_n: fem.Function,
    t_n: float,
    output_files: list[io.VTKFile],  # type: ignore
    results_dir: Path,
    context: context_t,
    variables: variables_t,
    postpro_exprs: expressions_t,
    mesh: Mesh,
) -> None:
    assert timesteps_metadata.timesteps is not None
    assert len(output_files) == len(variables) + len(postpro_exprs)
    timesteps_metadata.timesteps.append(t_n)

    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "timesteps.json", "w") as fh:
        json.dump(dataclasses.asdict(timesteps_metadata), fh, indent=4)

    for output_idx, output_file in enumerate(output_files[: len(variables)]):
        output_file.write_function(u_n.sub(output_idx).collapse(), t_n)

    export_postpro_exprs_impl(
        postpro_exprs,
        context,
        mesh,
        output_files[len(variables) :],
        t_n,
    )


def build_irk_tableau(settings: mdl.SolverSettings) -> bt.ButcherTableau:
    if settings.runge_kutta_method == mdl.RungeKuttaMethod.LOBATTO_IIIA.value:
        bt_cls = bt.LobattoIIIA
    elif settings.runge_kutta_method == mdl.RungeKuttaMethod.GAUSS_LEGENDRE.value:
        bt_cls = bt.GaussLegendre
    elif settings.runge_kutta_method == mdl.RungeKuttaMethod.RADAU_IIA.value:
        bt_cls = bt.RadauIIA
    else:
        raise ValueError(f"Unhandled Runge-Kutta method: {settings.runge_kutta_method}")

    return bt_cls(settings.num_time_stepping_stages)


def initialise_problem(
    mesh_data: MeshData, results_dir: Path, eqs_mdl: list[mdl.Equation]
) -> FenicsxProblem:

    problem = FenicsxProblem()
    problem.mesh_data = mesh_data
    problem.results_dir = results_dir

    problem.element = mixed_element_from_model(mesh_data.mesh, eqs_mdl)
    problem.function_space = fem.FunctionSpace(mesh_data.mesh, problem.element)

    problem.time = ufl.variable(ufl.Constant(mesh_data.mesh))
    problem.time_step = ufl.variable(ufl.Constant(mesh_data.mesh))

    problem.x = ufl.SpatialCoordinate(mesh_data.mesh)
    problem.ds = ufl.Measure(
        "ds", domain=mesh_data.mesh, subdomain_data=mesh_data.patch_tags
    )
    problem.dx = ufl.Measure(
        "dx", problem.mesh_data.mesh, subdomain_data=problem.mesh_data.domain_tags
    )

    problem.variables = {}
    for eq_idx, sub_trial_func in enumerate(ufl.TrialFunctions(problem.function_space)):  # type: ignore
        problem.variables[eqs_mdl[eq_idx].variable_name] = sub_trial_func  # type: ignore

    problem.test_variables = {}
    for eq_idx, sub_trial_func in enumerate(ufl.TestFunctions(problem.function_space)):  # type: ignore
        problem.test_variables[eqs_mdl[eq_idx].variable_name + "_test"] = sub_trial_func  # type: ignore

    return problem


def intialise_context(problem: FenicsxProblem) -> context_t:
    assert problem.mesh_data is not None
    return dict(
        x=problem.x,
        dx=problem.dx,
        ds=problem.ds,
        _mesh_=problem.mesh_data.mesh,
        time=problem.time,
        _time_step_=problem.time_step,
        ufl=ufl,
        pi=math.pi,
    )


def intialise_progress_bar(solver_settings: mdl.SolverSettings) -> tuple[Tqdm, float]:

    progress_bar = Tqdm(
        total=solver_settings.duration,
        bar_format="{desc}: {percentage:.3f}%|{bar}| [{elapsed}<{remaining}",
        file=TqdmLogger(get_logger()),
        leave=True,
    )
    t_last_update = solver_settings.start_time

    return progress_bar, t_last_update


def update_progress_bar(
    progress_bar: Tqdm, t_num: float, t_last_update: float, time_step_idx: float
) -> float:
    progress_bar.update(t_num - t_last_update)
    progress_bar.desc = f"{time_step_idx}, t: {t_num: .4e}"

    t_last_update = t_num

    return t_last_update


def form_from_initial_condition(
    eq: mdl.Equation, context: context_t, global_expressions: expressions_t
) -> fem.Function:

    ic_context = context.copy()
    add_expressions_to_context(ic_context, global_expressions)

    if eq.variable_shape == mdl.VariableType.SCALAR.value:
        form = to_scalar_form(eq.initial_condition, ic_context)
    elif eq.variable_shape == mdl.VariableType.VECTOR.value:
        form = to_vector_form(eq.initial_condition, ic_context)
    else:
        raise ValueError(f"Unexpected eq variable shape: {eq.variable_shape}")

    return form  # type: ignore


def initialise_solution(
    problem: FenicsxProblem,
    eqs_mdl: list[mdl.Equation],
    context: context_t,
    global_expressions: expressions_t,
    start_time: float,
) -> fem.Function:

    u_n_func: fem.Function = fem.Function(problem.function_space)  # type: ignore

    for idx, eq in enumerate(eqs_mdl):

        ic_form = form_from_initial_condition(eq, context, global_expressions)

        assert problem.time is not None
        expr = fem.Expression(
            ufl.replace(ic_form, {problem.time: start_time}), problem.function_space.sub(idx).element.interpolation_points()  # type: ignore
        )
        u_n_func.sub(idx).interpolate(expr)

    return u_n_func


def solve_irk_problem(
    solver_settings: mdl.SolverSettings,
    mesh_data: MeshData,
    eqs_mdl: list[mdl.Equation],
    global_expressions: expressions_t,
    postpro_expressions: expressions_t,
    results_dir: Path,
) -> bool:

    # For each sub-space p:
    #
    #   a(u⁽ᵖ⁾,w⁽ᵖ⁾) = L(v⁽ᵖ⁾) + ∫ ∂u⁽ᵖ⁾/∂t.w⁽ᵖ⁾.dx, ∀ w⁽ᵖ⁾ ∈ V [R-1]
    #
    # For compactness of what follows we may define the set of functionals R⁽ᵖ⁾:
    #
    #   R⁽ᵖ⁾(t, x, u⁽ᵖ⁾, w⁽ᵖ⁾) := a(u⁽ᵖ⁾,w⁽ᵖ⁾) - L(v⁽ᵖ⁾) [R-2]
    #
    # So that trivially:
    #
    #   R⁽ᵖ⁾(t, x, u⁽ᵖ⁾, w⁽ᵖ⁾) = ∫ ∂u⁽ᵖ⁾/∂t.w⁽ᵖ⁾.dx, ∀ w⁽ᵖ⁾ ∈ V [R-3]
    #
    # At the i-th IRK time point we define the field kᵢ⁽ᵖ⁾ as:
    #
    #   kᵢ⁽ᵖ⁾ := ∂u⁽ᵖ⁾/∂t(t=tᵢ)  [R-4]
    #
    # and the IRK estimate of u⁽ᵖ⁾(tᵢ) is:
    #
    #   uᵢ⁽ᵖ⁾ := u⁽ᵖ⁾(t=tᵢ) = u⁽ᵖ⁾(t=tₙ) + hAᵢⱼkⱼ⁽ᵖ⁾ [R-5]
    #
    # These kᵢ⁽ᵖ⁾ will be our problem variables and FenicsX wants a compatible
    # set of test variables to enforce the equality relationship (∀ w⁽ᵖ⁾ ∈ V).
    # Thus at each time point we will use vᵢ⁽ᵖ⁾ to refer to our test function.
    #
    # thus [R-3] becomes:
    #
    #  Rᵢ⁽ᵖ⁾ = ∫ kᵢ⁽ᵖ⁾.vᵢ⁽ᵖ⁾.dx, ∀ vᵢ⁽ᵖ⁾ ∈ V [R-6]
    #
    # where Rᵢ⁽ᵖ⁾ is the result of substituting: t=tᵢ and u⁽ᵖ⁾ = uᵢ⁽ᵖ⁾ (eq
    # [R-5]) into R⁽ᵖ⁾.
    #
    # Combining the functionals for each subspace yields:
    #
    #   Rᵢ = Σₚ Rᵢ⁽ᵖ⁾ = Σₚ ∫ kᵢ⁽ᵖ⁾.vᵢ⁽ᵖ⁾.dx, ∀ vᵢ⁽ᵖ⁾ ∈ V [R-7]
    #
    #
    # We know the combined form for a and L so we can get:
    #
    #   Rᵢ = a(tᵢ, x, uᵢ⁽⁰⁾, ... , vᵢ⁽⁰⁾, ...) - L(tᵢ, x, vᵢ⁽⁰⁾, ...)
    #
    # Thus to form the Rᵢ functionals we need only form R via eq. [R-2] then
    # substitute for t=tᵢ, uᵢ⁽ᵖ⁾ (via eq. [R-5]) and vᵢ⁽ᵖ⁾=w⁽ᵖ⁾.

    do_display_progress = MPI.COMM_WORLD.rank == 0
    t_last_update = 0
    progress_bar = None

    problem = initialise_problem(mesh_data, results_dir=results_dir, eqs_mdl=eqs_mdl)
    assert problem.mesh_data is not None
    assert problem.variables is not None
    assert problem.function_space is not None

    context = intialise_context(problem)

    tableau = build_irk_tableau(solver_settings)

    #
    # Initialise:
    #

    # Function Space for the kᵢ⁽ᵖ⁾ := ∂u⁽ᵖ⁾/∂t(t=tᵢ)  [R-4]

    function_space_stages = fem.FunctionSpace(
        mesh_data.mesh, ufl.MixedElement(tableau.num_stages * [problem.element])
    )

    problem.u = fem.Function(function_space_stages)  # type: ignore

    if solver_settings.solver_type == mdl.SolverType.LINEAR.value:
        k_trial_stages = ufl.TrialFunctions(function_space_stages)
    elif solver_settings.solver_type == mdl.SolverType.NON_LINEAR.value:
        k_trial_stages = ufl.split(problem.u)
    else:
        raise ValueError(f"Unexpected solver_type: {solver_settings.solver_type}")

    k_test_stages = ufl.TestFunctions(function_space_stages)

    #

    u_n_func = initialise_solution(
        problem, eqs_mdl, context, global_expressions, solver_settings.start_time
    )

    # bcs: constrain V_stages to boundary value of ∂u/∂t
    # We will need to convert the users dirichlet conditions to their derivative
    # for use for each kᵢ

    info_log("Initialising boundary conditions", True)

    bc_context = context.copy()
    add_expressions_to_context(bc_context, global_expressions)

    problem.dirichlet_conditions = []

    # NOTE(CR) We go eq->stage but sub-spaces are reverse ordered sub(stage_idx).sub(eq_idx)
    u_bcs_step_at_start_time = []  # [sub-space/eq][bc]
    k_bcs_form: list[list[ufl_Expr]] = []  # [sub-space/eq][bc]
    k_bcs_funcs: list[
        list[list[fem.Function]]
    ] = []  # [sub-space/eq][bc][stage] func_bc_n_stage_m = k_bcs_funcs[n][m][p]

    for eq_idx, eq_mdl in enumerate(eqs_mdl):

        info_log(
            f"Defining dirichlet conditions for: {eq_mdl.variable_name}",
            rank0_only=True,
        )

        u_eq_bcs_step_at_start_time = []
        u_bcs_step_at_start_time.append(u_eq_bcs_step_at_start_time)

        k_eq_bcs_form = []
        k_bcs_form.append(k_eq_bcs_form)

        k_eq_bcs_funcs = []
        k_bcs_funcs.append(k_eq_bcs_funcs)

        for dir_bc_mdl in eq_mdl.dirichlet_conditions:
            k_eq_bc_func_stages = []

            info_log(
                f"Defining bc: {dir_bc_mdl.value} on patches: {dir_bc_mdl.domain_indices}",
                rank0_only=True,
            )

            u_dir_bc_form = to_scalar_form(dir_bc_mdl.value, bc_context)
            k_eq_bcs_form.append(expand_derivatives(ufl.diff(u_dir_bc_form, problem.time)))  # type: ignore

            # user might not have specified a self consistent boundary condition and initial condition
            # we apply the discontinuity smoothly over the first time step

            u_eq_bcs_step_at_start_time.append(
                ufl.replace(
                    u_dir_bc_form - form_from_initial_condition(eq_mdl, context, global_expressions),  # type: ignore
                    {problem.time: fem.Constant(mesh_data.mesh, default_scalar_type(solver_settings.start_time))},  # type: ignore
                ),
            )

            for stage in range(tableau.num_stages):

                sub_space = function_space_stages.sub(stage).sub(eq_idx)
                collapsed_sub_space, _ = sub_space.collapse()

                k_eq_bc_func_stages.append(fem.Function(collapsed_sub_space))

                for facet_index in dir_bc_mdl.domain_indices:
                    dofs = fem.locate_dofs_topological(
                        (sub_space, collapsed_sub_space),
                        mesh_data.mesh.topology.dim - 1,
                        mesh_data.patch_tags.find(facet_index),
                    )

                    bc_mdl = fem.dirichletbc(
                        k_eq_bc_func_stages[stage], dofs, sub_space
                    )
                    problem.dirichlet_conditions.append(bc_mdl)

            k_eq_bcs_funcs.append(k_eq_bc_func_stages)

    # Forms for u(tᵢ) [R-5]

    A_form = ufl.as_matrix(tableau.A)
    u_form_stages = []  # [sub-space/eq_idx][stage]
    for eq_idx, _ in enumerate(eqs_mdl):
        k_eq_trial_form = ufl.as_vector(
            [ufl.split(k_trial_stage)[eq_idx] for k_trial_stage in k_trial_stages]
        )

        u_form_stages.append(
            ufl.as_vector([u_n_func.sub(eq_idx)] * tableau.num_stages)
            + problem.time_step * A_form * k_eq_trial_form  # type: ignore
        )

    # R for each stage

    a_form, L_form = a_L_from_equations(problem, eqs_mdl, global_expressions, context)

    R_form = a_form - L_form

    # TODO(CR) continue from here ...
    R_form_stages = []

    for stage in range(tableau.num_stages):
        # NOTE(CR) We must wait to do the explicit time replacement for the main loop since this changes for each iteration
        # NOTE(CR) We must replace the test variables in a and L with those of the IRK problem

        R_form_stage = R_form

        for eq_idx, eq_mdl in enumerate(eqs_mdl):
            R_form_stage = ufl.replace(
                R_form_stage,
                {
                    problem.variables[eq_mdl.variable_name]: u_form_stages[eq_idx][stage],  # type: ignore - it is hashable
                    problem.test_variables[f"{eq_mdl.variable_name}_test"]: ufl.split(k_test_stages[stage])[eq_idx],  # type: ignore - it is hashable
                },
            )

        R_form_stages.append(R_form_stage)

    # Prepare error estimate

    equals_one = to_function(
        fem.Constant(mesh_data.mesh, default_scalar_type(1)),
        problem.function_space.sub(0).collapse()[0],
    )
    measure = numpy.real(
        mesh_data.mesh.comm.allreduce(
            fem.assemble_scalar(fem.form(equals_one * problem.dx())),  # type: ignore
            op=MPI.SUM,
        )
    )

    # Prepare Output

    timesteps_metadata = mdl.TimestepsMetadata(timesteps=[])

    assert problem.results_dir is not None

    eqs_vars = [eq.variable_name for eq in eqs_mdl]
    pp_vars = [pp_variable for pp_variable in postpro_expressions]

    with ExitStack() as stack:

        output_files = [
            stack.enter_context(
                io.VTKFile(  # pyright: ignore[reportPrivateImportUsage]
                    mesh_data.mesh.comm, problem.results_dir / f"{var_name}.vtu", "w"
                )
            )
            for var_name in eqs_vars + pp_vars
        ]

        export_context = context.copy()
        sub_funcs = ufl.split(u_n_func)
        for index, variable in enumerate(problem.variables):
            export_context[variable] = sub_funcs[index]
        add_expressions_to_context(export_context, global_expressions)

        export_time_step(
            timesteps_metadata,
            u_n_func,
            solver_settings.start_time,
            output_files,
            problem.results_dir,
            export_context,
            problem.variables,
            postpro_expressions,
            problem.mesh_data.mesh,
        )

        #
        # Perform time steps
        #

        t_num = solver_settings.start_time
        dt_num = solver_settings.initial_time_step_size
        time_step_idx = 0
        rms_error = -1

        if do_display_progress:
            progress_bar, t_last_update = intialise_progress_bar(solver_settings)

        while t_num < solver_settings.duration + solver_settings.start_time:
            if progress_bar is not None:
                t_last_update = update_progress_bar(
                    progress_bar, t_num, t_last_update, time_step_idx
                )

            # Calculate sub-step times: tᵢ

            t_stages_num = t_num + dt_num * cast(numpy.ndarray, tableau.c)
            t_stages_as_constant = [
                fem.Constant(mesh_data.mesh, default_scalar_type(t_stage_num))
                for t_stage_num in t_stages_num.tolist()
            ]  # important to wrap value as fem.Constant to avoid re-compilation when the value changes

            dt_as_constant = fem.Constant(mesh_data.mesh, default_scalar_type(dt_num))

            # bcs: update boundary functions for this time step's sub-steps
            for eq_idx, _ in enumerate(eqs_mdl):

                k_eq_bcs_form = k_bcs_form[eq_idx]
                k_eq_bcs_funcs = k_bcs_funcs[eq_idx]
                u_eq_bcs_step_at_start_time = u_bcs_step_at_start_time[eq_idx]

                for bc_idx, k_eq_bc_form in enumerate(k_eq_bcs_form):
                    k_eq_bc_func_stages = k_eq_bcs_funcs[bc_idx]
                    u_eq_step_at_start_time = u_eq_bcs_step_at_start_time[bc_idx]

                    for stage in range(tableau.num_stages):
                        k_eq_bc_at_t = ufl.replace(
                            k_eq_bc_form, {problem.time: t_stages_as_constant[stage]}  # type: ignore
                        ) + fem.Constant(mesh_data.mesh, default_scalar_type(0))

                        if time_step_idx == 0:
                            k_eq_bc_at_t += u_eq_step_at_start_time / dt_num

                        assert problem.function_space is not None
                        k_eq_bc_at_t_expr = fem.Expression(
                            k_eq_bc_at_t,
                            problem.function_space.sub(
                                eq_idx
                            ).element.interpolation_points(),
                        )
                        k_eq_bc_func_stages[stage].interpolate(k_eq_bc_at_t_expr)

            # Finally form problem for this time step [R-7]: Σₚ Rᵢ⁽ᵖ⁾ = Σₚ ∫ kᵢ⁽ᵖ⁾.vᵢ⁽ᵖ⁾.dx

            assert problem.dx is not None

            F = ufl_Zero()

            for stage in range(tableau.num_stages):
                F_stage = (  # type: ignore
                    ufl.inner(k_trial_stages[stage], k_test_stages[stage])
                    * problem.dx()
                    - R_form_stages[stage]
                )
                F += ufl.replace(
                    F_stage,
                    {
                        problem.time: t_stages_as_constant[stage],  # type: ignore
                        problem.time_step: dt_as_constant,  # type: ignore
                    },
                )

            problem.a = ufl.lhs(F)
            problem.L = ufl.rhs(F)

            converged = solve_stationary_problem(
                problem, solver_settings, verbose=False
            )
            if not converged:
                raise RuntimeError("Time step failed to converge")

            # calculate value at end of time step

            def estimate_next_u_n(b):
                b_transpose_form = ufl.as_matrix([b.tolist()])

                u_n_plus_1_func: fem.Function = fem.Function(problem.function_space)  # type: ignore
                for idx, _ in enumerate(eqs_mdl):
                    k_eq_stages_form = ufl.as_vector(
                        [ufl.split(ki)[idx] for ki in ufl.split(problem.u)]
                    )
                    u_n_eq_plus_1_form = (
                        u_n_func.sub(idx)
                        + (dt_as_constant * b_transpose_form * k_eq_stages_form)[0]  # type: ignore
                    )
                    u_n_eq_plus_1_expr = fem.Expression(
                        u_n_eq_plus_1_form,
                        problem.function_space.sub(idx).element.interpolation_points(),  # type: ignore
                    )
                    u_n_plus_1_func.sub(idx).interpolate(
                        u_n_eq_plus_1_expr
                    )  # pyright: ignore[reportOptionalMemberAccess]

                return u_n_plus_1_func

            u_n_plus_1_func = estimate_next_u_n(tableau.b)
            utilde_n_plus_1_func = estimate_next_u_n(tableau.btilde)

            # estimate error

            error = numpy.real(
                mesh_data.mesh.comm.allreduce(
                    fem.assemble_scalar(
                        fem.form(ufl.inner(utilde_n_plus_1_func - u_n_plus_1_func, utilde_n_plus_1_func - u_n_plus_1_func) * problem.dx())  # type: ignore
                    ),
                    op=MPI.SUM,
                )
            )

            rms_error = numpy.sqrt(error / measure)

            # error_1 / error_2 = (h_1/h_2)^m

            if solver_settings.adapt_time_step:

                # NOTE(CR) We will still used our best estimate for u_n but will pretend for the
                # case of step size selection that we are using our b_tilde estimate.

                est_step_size = (
                    0.9  # slightly underestimate the step size to avoid rejections
                    * dt_num
                    * (solver_settings.target_error / (rms_error + 1e-9))
                    ** (1.0 / (tableau.order - 1))
                )  ## assume order of btilde always one less?

                if time_step_idx < 2:
                    dt_new_num = est_step_size
                else:
                    dt_new_num = min(
                        solver_settings.max_time_step,
                        (1 - SMOOTHING_FACTOR) * dt_num
                        + SMOOTHING_FACTOR * est_step_size,
                    )

                if (
                    time_step_idx > 0 and rms_error > solver_settings.target_error
                ):  # first step will be weird because of possible bc jump
                    dt_num = dt_new_num
                    continue  # try again ...

            else:
                dt_new_num = dt_num

            # advance to the next time step

            u_n_func.x.array[:] = u_n_plus_1_func.x.array
            t_num = t_num + dt_num
            dt_num = dt_new_num
            time_step_idx += 1

            export_time_step(
                timesteps_metadata,
                u_n_func,
                t_num,
                output_files,
                problem.results_dir,
                export_context,
                problem.variables,
                postpro_expressions,
                problem.mesh_data.mesh,
            )
    if progress_bar is not None:
        progress_bar.update(solver_settings.duration - t_last_update)
        progress_bar.close()

    return True
