import dataclasses
import json
import logging
import shutil
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, cast

import fenics.model.equation as eq
import fenics.model.solver_settings as solver_settings
import XCore
import XCoreModeling as xm
import XPostProPython as pp
from s4l_core.simulator_plugins.base.model.simulation_base import SimulationBase
from s4l_core.simulator_plugins.base.solver.driver.api_models import ApiSimulationBase
from s4l_core.simulator_plugins.base.solver.project_runner import SolverBackend, config_type_t
from fenics.model.expressions import (
    GlobalExpressions,
    PostProcessingExpressions,
)
from fenics.model.utils import number_with_suffix
from fenics.solver.driver import api_models

if TYPE_CHECKING:
    from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem

logger = logging.getLogger(__name__)

domain_id_map_t = dict[str, int]  # map from uuid (as str) to the domain id


def get_source_spec()->Path:
    src_spec = find_spec("fenics.solver.driver")
    assert src_spec is not None
    assert src_spec.origin is not None
    
    return Path(src_spec.origin).parent

class FenicsSimulation(SimulationBase):
    """Concrete implementation of the Simulation class."""

    @classmethod
    def get_simulation_type_name(cls) -> str:
        return "Fenics"

    @classmethod
    def get_simulation_description(cls) -> str:
        return "Simulates anything using FEncis."

    @classmethod
    def get_simulation_icon(cls) -> str:
        return "icons/XSimulatorUI/new_simulation.ico"

    def __init__(
        self,
        parent: "TreeItem",
        sim_desc: str = "Simulation",
        sim_notes: str = "",
    ) -> None:
        super().__init__(parent, sim_desc, sim_notes)

        self._simulation_mesh_id: str | None = None

    def on_initialize_settings(self) -> None:
        """Initialize all simulation settings."""

        simulation_type = XCore.PropertyEnum(
            [
                api_models.SimulationType.STATIONARY.value,
                api_models.SimulationType.EIGENVALUE.value,
                api_models.SimulationType.TIME_DOMAIN.value,
            ],
            0,
        )
        simulation_type.Description = "Simulation Type"
        self._properties.Add("simulation_type", simulation_type)

        self._equations: "eq.Equations" = eq.Equations(self)
        self._solver_settings = solver_settings.SolverSettings(self)
        self._global_expressions = GlobalExpressions(self)
        self._post_processing_expressions = PostProcessingExpressions(self)

    @property
    def simulation_type_prop(self) -> XCore.PropertyEnum:
        prop = self._properties.simulation_type
        assert isinstance(prop, XCore.PropertyEnum)
        return prop

    @property
    def simulation_mesh(self) -> xm.UnstructuredMesh | None:
        if self._simulation_mesh_id is None:
            return None

        entity = xm.GetActiveModel().LookupEntity(XCore.Uuid(self._simulation_mesh_id))
        assert isinstance(entity, xm.UnstructuredMesh)
        return entity

    @property
    def equations(self) -> "eq.Equations":
        return self._equations

    @property
    def solver_settings(self) -> solver_settings.SolverSettings:
        return self._solver_settings

    @property
    def global_expressions(self) -> GlobalExpressions:
        return self._global_expressions

    @property
    def post_processing_expressions(self) -> PostProcessingExpressions:
        return self._post_processing_expressions

    def register_extractor(self) -> pp.PythonModuleAlgorithm:
        """Create the appropriate extractor for this simulation type."""
        return pp.PythonModuleAlgorithm(
            "fenics.model.simulation_extractor",
            0,
            1,
        )

    def _create_project_config(self ) -> config_type_t:
        solver_settings = self.solver_settings.as_api_model()
        num_processes, field_type = solver_settings.num_processes, solver_settings.field_type.lower()
        assert field_type in {"real", "complex"}

        source_script = f"source /usr/local/bin/dolfin-{field_type}-mode"
        if num_processes == 1:
            mpi_prefix = ""
        else:
            mpi_prefix = f"/opt/conda/bin/mpirun -np {num_processes} "

        solver_dir = get_source_spec()

        inputs_dir = self.project_root / self.results_dir / "input_files"
        outputs_dir = self.project_root / self.results_dir / "output_files"

        solver_enty_point = solver_dir / "main.py"

        if not solver_enty_point.is_file():
            raise FileNotFoundError(
                f"Solver entry point not found: {solver_enty_point}"
            )
        
        config = {
            "cmd" : [source_script, "&&", mpi_prefix, "/opt/conda/bin/python3", str(solver_enty_point), "-i", str(inputs_dir), "-o", str(outputs_dir)],
            "cwd:" : "/fenics_driver",
            "env" : {"OMP_NUM_THREADS" : "1"}
        }

        return config        

    def solver_backend(self) -> tuple[SolverBackend, config_type_t | None]:
        return SolverBackend.PROCESS, self._create_project_config()

    def _prepare_inputs(self) -> Path:
        logger.info(f"Running problem in: {self.project_root / self.results_dir}")

        inputs_dir = self.project_root / self.results_dir / "input_files"
        if Path(inputs_dir).is_dir():
            shutil.rmtree(inputs_dir)

        inputs_dir.mkdir(exist_ok=True, parents=True)

        vtu_path = (inputs_dir / "mesh.vtu").resolve()

        exporter = xm.CreateExporterFromFile(str(vtu_path))
        exporter.Options.ExportDomainNames.Value = (
            False  # meshio can't cope with the String field
        )

        geom_prop = exporter.Options.GeometryToExport
        assert isinstance(geom_prop, XCore.PropertyEnum)
        geom_prop.Value = 2
        assert geom_prop.ValueDescription == "Domains+Patches"

        unit_prop = exporter.Options.LengthUnit
        assert isinstance(unit_prop, XCore.PropertyEnum)
        unit_prop.Value = 1
        assert unit_prop.ValueDescription == "m"

        exporter.Export([self.simulation_mesh], str(vtu_path))

        with open(inputs_dir / "input_file.json", "w") as fh:
            json.dump(dataclasses.asdict(self.as_api_model()), fh, indent=4)

        return get_source_spec()

    def clear_status_recursively(self) -> None:
        super().clear_status_recursively()
        self.clear_status()
        self.solver_settings.clear_status()
        self._equations.clear_status_recursively()

    def validate(self) -> bool:
        self.clear_status_recursively()

        if len(self._equations.elements) == 0:
            self.status_icons = [
                "icons/TaskManager/Warning.ico",
            ]
            validation_error = "No equations defined for this simulation"
            self.status_icons_tooltip += f"{validation_error}.  "
            logger.error(f"{self.description}: {validation_error}")
            return False

        ok = True

        # Check exactly one equation since eigenvalue problems don't support coupling yet TODO(CR) fix this!

        if self.simulation_type_prop.ValueDescription in (
            api_models.SimulationType.EIGENVALUE.value,
        ):
            if len(self._equations.elements) != 1:  # TODO
                self.status_icons = [
                    "icons/TaskManager/Warning.ico",
                ]
                validation_error = f"For the {self.simulation_type_prop.ValueDescription} simulation type there cannot be more than 1 equation defined"
                self.status_icons_tooltip += f"{validation_error}.  "
                logger.error(f"{self.description}: {validation_error}")
                return False

        # Check variable names are unique

        variable_names: dict[str, str] = dict()
        for idx, equation in enumerate(self._equations.elements):
            if equation.variable_name in variable_names:
                self.status_icons = [
                    "icons/TaskManager/Warning.ico",
                ]
                validation_error = f"{number_with_suffix(idx+1)} equation '{equation.description}' has a duplicated variable name: {equation.variable_name} (it also appears in {variable_names[equation.variable_name]})"
                self.status_icons_tooltip += f"{validation_error}.  "
                logger.error(f"{self.description}: {validation_error}")
                ok = False

            variable_names[equation.variable_name] = equation.description

        # Check equations

        for idx, equation in enumerate(self._equations.elements):
            result = equation.validate()
            ok = ok and result

        if not ok:
            return False

        # Check consistent meshes

        for idx, equation in enumerate(self._equations.elements):
            assert (
                equation.unstructured_mesh_id is not None
            )  # should not happen if validation succeeded

            if idx == 0:
                self._simulation_mesh_id = equation.unstructured_mesh_id
            elif self._simulation_mesh_id != equation.unstructured_mesh_id:
                self.status_icons = [
                    "icons/TaskManager/Warning.ico",
                ]
                validation_error = f"Equation {idx} ({equation.variable_name}) is applied to a different mesh to those of the earlier equations in this simulation"
                self.status_icons_tooltip += f"{validation_error}.  "
                logger.error(f"{self.description}: {validation_error}")

                return False
            else:
                pass  # all good

        # Check solver settings:

        ok = ok and self.solver_settings.validate()

        ## e.g. no MPI + Iterative + ILU

        return ok

    def as_api_model(self) -> ApiSimulationBase:
        if not self.validate():
            raise RuntimeError("Validation failed")

        api_equations: list[api_models.Equation] = []
        for equation in self._equations.elements:
            api_equations.append(equation.as_api_model())

        result = api_models.Simulation(
            simulation_type=self.simulation_type_prop.ValueDescription,
            equations=api_equations,
            global_expressions=self._global_expressions.expressions,
            post_processing_expressions=self._post_processing_expressions.expressions,
            solver_settings=self._solver_settings.as_api_model(),
        )
        return cast(
            ApiSimulationBase, result
        )  # Cast to the base class type for compatibility

    def get_solver_src(self) -> str:
        """Get the module path to the solver driver."""
        return "fenics.solver.driver"
