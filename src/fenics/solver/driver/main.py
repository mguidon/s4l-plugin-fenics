#!usr/bin/python3

import os
import sys
from pathlib import Path

import core_utils

WORK_DIR = Path("/work")
INPUT_DIR = WORK_DIR / "input_files"
OUTPUT_DIR = WORK_DIR / "output_files"

INPUT_FILE = INPUT_DIR / "input_file.json"
MESH_DOMAINS_PTH = INPUT_DIR / "mesh-domains.vtu"
MESH_PATCHES_PTH = INPUT_DIR / "mesh-patches.vtu"

core_utils.init_results_dir(OUTPUT_DIR)
LOGGER = core_utils.init_logging(OUTPUT_DIR)

import api_models as mdl  # noqa: E402
import dolfinx  # noqa: E402
import dolfinx.cpp.log as fn_log  # noqa: E402
import simulation_utils as ut  # noqa: E402
from irk_utils import solve_irk_problem  # noqa: E402
from mpi4py import MPI  # noqa: E402

RANK = MPI.COMM_WORLD.rank

fn_log.set_log_level(fn_log.LogLevel.WARNING)
fn_log.set_output_file(str(OUTPUT_DIR / f"fenicsx_log_rank{RANK}.log"))


def main() -> tuple[ut.FenicsxProblem, bool]:

    ut.info_log(
        f"dolfinx version: {dolfinx.__version__}",  # pyright: ignore[reportPrivateImportUsage]
        True,
    )
    ut.info_log(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}", True)

    # Problem definitions

    with open(INPUT_DIR / "input_file.json") as fh:
        simulation = mdl.Simulation.schema().loads(fh.read())  # type: ignore

    if not MESH_DOMAINS_PTH.is_file():
        raise ValueError(f"Failed to find: {MESH_DOMAINS_PTH}")

    if not MESH_PATCHES_PTH.is_file():
        raise ValueError(f"Failed to find: {MESH_PATCHES_PTH}")

    problem = ut.FenicsxProblem()
    problem.results_dir = OUTPUT_DIR

    # Mesh

    with ut.TimingContext("Importing mesh"):
        problem.mesh_data = ut.import_s4l_mesh(
            MESH_DOMAINS_PTH, MESH_PATCHES_PTH, problem.results_dir
        )

    # if MPI.COMM_WORLD.size == 1:
    #     problem.mesh_data.display_mesh(
    #         False, save_path=problem.results_dir / "mesh.pdf"
    #     )
    # else:
    #     ut.info_log("To preview imported mesh run on a single process", True)

    # Form Definition

    if simulation.simulation_type == mdl.SimulationType.TIME_DOMAIN.value:

        with ut.TimingContext("Solving time domain problem"):

            ok = solve_irk_problem(
                simulation.solver_settings,
                problem.mesh_data,
                simulation.equations,
                simulation.global_expressions,
                simulation.post_processing_expressions,
                OUTPUT_DIR,
            )

    else:

        with ut.TimingContext("Defining function space"):
            problem.function_space = ut.function_space_from_model(
                problem.mesh_data.mesh, simulation
            )

        with ut.TimingContext("Defining PDE's"):
            ut.init_problem_variables(problem, simulation)
            problem.a, problem.L = ut.a_L_from_model(problem, simulation)

        with ut.TimingContext("Defining Dirichlet boundary conditions"):
            problem.dirichlet_conditions = ut.dirichlet_conditions_from_model(
                problem, simulation
            )

        # Solve

        with ut.TimingContext("Solving PDE's"):
            fn_log.set_log_level(fn_log.LogLevel.INFO)
            ok = ut.solve_problem(problem, simulation)
            fn_log.set_log_level(fn_log.LogLevel.WARNING)

        # Export

        with ut.TimingContext("Exporting Solution"):
            ut.export_solved_problem(problem, simulation)

        with ut.TimingContext("Exporting Post-processing expressions"):
            ut.export_postpro_exprs(
                problem,
                simulation,
            )

    # Finalize

    if not ok:
        LOGGER.error("❌ Failed to solve problem ❌")
    else:
        ut.info_log("✨ Finished solving problem ✨", True)

    return problem, ok


if __name__ == "__main__":
    problem, ok = main()
    sys.exit(0 if ok else 1)
