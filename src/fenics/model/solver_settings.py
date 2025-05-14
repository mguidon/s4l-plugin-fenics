import asyncio
import logging

import fenics.model.fenics_simulation as sim
import XCore
import XCoreHeadless
from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem
from fenics.solver.driver import api_models

logger = logging.getLogger(__name__)


class SolverSettings(TreeItem):
    def __init__(
        self,
        parent: TreeItem,
    ) -> None:
        super().__init__(parent, icon="icons/XSimulatorUI/SolverSettings.ico")

        self._properties = XCoreHeadless.DialogOptions()
        self._properties.Description = "Solver Settings"

        field_type = XCore.PropertyEnum(
            (api_models.FieldType.REAL.value, api_models.FieldType.COMPLEX.value), 0
        )
        field_type.Description = "Field Type"
        self._properties.Add("field_type", field_type)

        self._properties.Add("num_processes", self._create_num_processes_prop())

        #

        self._add_shared_sim_props()

        ksp_props = XCoreHeadless.DialogOptions()
        ksp_props.Description = "Iterative Solver Settings"
        self._properties.Add("ksp", ksp_props)
        self._add_ksp_props()

        eigenvalue_props = XCoreHeadless.DialogOptions()
        eigenvalue_props.Description = "Eigenvalue Solver Settings"
        self._properties.Add("eigenvalue", eigenvalue_props)
        self._add_eigenvalue_props()

        time_domain_props = XCoreHeadless.DialogOptions()
        time_domain_props.Description = "Time Domain Solver Settings"
        self._properties.Add("time_domain", time_domain_props)
        self._add_time_stepping_props()

        #

        self._connect_signals()

    def _add_shared_sim_prop(self, prop_name, prop: XCore.Property) -> None:
        self._properties.Add(prop_name, prop)

    @property
    def _ksp_props(self) -> XCoreHeadless.DialogOptions:
        ksp_prop_group = self._properties.ksp
        assert isinstance(ksp_prop_group, XCoreHeadless.DialogOptions)
        return ksp_prop_group

    def _add_ksp_prop(self, prop_name, prop: XCore.Property) -> None:
        self._ksp_props.Add(prop_name, prop)

    @property
    def _eigenvalue_props(self) -> XCoreHeadless.DialogOptions:
        eigenvalue_prop_group = self._properties.eigenvalue
        assert isinstance(eigenvalue_prop_group, XCoreHeadless.DialogOptions)
        return eigenvalue_prop_group

    def _add_eigenvalue_sim_prop(self, prop_name, prop: XCore.Property) -> None:
        self._eigenvalue_props.Add(prop_name, prop)

    @property
    def _time_domain_props(self) -> XCoreHeadless.DialogOptions:
        time_domain_prop_group = self._properties.time_domain
        assert isinstance(time_domain_prop_group, XCoreHeadless.DialogOptions)
        return time_domain_prop_group

    def _add_time_stepping_sim_prop(self, prop_name, prop: XCore.Property) -> None:
        self._time_domain_props.Add(prop_name, prop)

    def _add_shared_sim_props(self) -> None:

        self._add_shared_sim_prop("solver_type", self._create_solver_type_prop())

        self._add_shared_sim_prop(
            "linear_solver_type", self._create_linear_solver_type_prop()
        )

        nonlin_rtol = XCore.PropertyReal(1e-8, 0, 1)
        nonlin_rtol.Description = "Relative Newton Solver Tolerance"
        self._add_shared_sim_prop("nonlin_rtol", nonlin_rtol)

        nonlin_atol = XCore.PropertyReal(1e-10, 0, 1)
        nonlin_atol.Description = "Absolute Newton Solver Tolerance"
        self._add_shared_sim_prop("nonlin_atol", nonlin_atol)

    def _add_ksp_props(self) -> None:

        self._add_ksp_prop("ksp_pc", self._create_ksp_pc_prop())

        self._add_ksp_prop("ksp_type", self._create_ksp_type_prop())

        self._add_ksp_prop("ksp_rtol", self._create_ksp_rtol_prop())

        self._add_ksp_prop("ksp_atol", self._create_ksp_atol_prop())

    def _add_eigenvalue_props(self) -> None:
        self._add_eigenvalue_sim_prop(
            "num_eigenvals", self._create_num_eigenvals_prop()
        )
        self._add_eigenvalue_sim_prop(
            "target_eigenval", self._create_target_eigenval_prop()
        )
        self._add_eigenvalue_sim_prop("eigenval_tol", self._create_eigenval_tol_prop())

    def _add_time_stepping_props(self) -> None:
        self._add_time_stepping_sim_prop("start_time", self._create_start_time_prop())
        self._add_time_stepping_sim_prop("duration", self._create_duration_prop())
        self._add_time_stepping_sim_prop(
            "runge_kutta_method", self._create_runge_kutta_method_prop()
        )
        self._add_time_stepping_sim_prop(
            "num_time_stepping_stages", self._create_num_time_stepping_stages_prop()
        )
        self._add_time_stepping_sim_prop(
            "initial_time_step_size", self._create_initial_time_step_size_prop()
        )
        self._add_time_stepping_sim_prop(
            "adapt_time_step", self._create_adapt_time_step_prop()
        )
        self._add_time_stepping_sim_prop(
            "max_time_step", self._create_max_time_step_prop()
        )
        self._add_time_stepping_sim_prop(
            "target_error", self._create_target_error_prop()
        )

    def _create_solver_type_prop(self) -> XCore.PropertyEnum:
        solver_type = XCore.PropertyEnum(
            [
                api_models.SolverType.LINEAR.value,
                api_models.SolverType.NON_LINEAR.value,
            ],
            0,
        )
        solver_type.Name = "solver_type"
        solver_type.Description = "Solver Type"
        return solver_type

    def _create_num_processes_prop(self) -> XCore.PropertyInt:
        num_processes = XCore.PropertyInt(1, 1, 32)
        num_processes.Name = "num_processes"
        num_processes.Description = "Number of Processes"
        return num_processes

    def _create_linear_solver_type_prop(self) -> XCore.PropertyEnum:
        linear_solver_type = XCore.PropertyEnum(
            [
                api_models.LinearSolverType.DIRECT_LU.value,
                api_models.LinearSolverType.ITERATIVE_KSP.value,
            ],
            0,
        )
        linear_solver_type.Name = "linear_solver_type"
        linear_solver_type.Description = "Linear System Solver"
        linear_solver_type.ToolTip = "Method used to solve linear system"
        return linear_solver_type

    def _create_ksp_pc_prop(self) -> XCore.PropertyEnum:
        ksp_pc = XCore.PropertyEnum(
            [
                api_models.KSPPreConditionerType.EUCLID.value,
                api_models.KSPPreConditionerType.PILUT.value,
                api_models.KSPPreConditionerType.BOOMER_AMG.value,
                api_models.KSPPreConditionerType.ILU.value,
                api_models.KSPPreConditionerType.NONE.value,
            ],
            0,
        )
        ksp_pc.Name = "ksp_pc"
        ksp_pc.Description = "KSP Preconditioner"
        return ksp_pc

    def _create_ksp_type_prop(self) -> XCore.PropertyEnum:
        ksp_type = XCore.PropertyEnum(
            [
                api_models.KSPType.CG.value,
                api_models.KSPType.GMRES.value,
            ],
            0,
        )
        ksp_type.Name = "ksp_type"
        ksp_type.Description = "KSP Method"

        return ksp_type

    def _create_ksp_rtol_prop(self) -> XCore.PropertyReal:
        ksp_rtol = XCore.PropertyReal(1e-6, 0, 1)
        ksp_rtol.Name = "ksp_rtol"
        ksp_rtol.Description = "KSP Relative Tolerance"
        return ksp_rtol

    def _create_ksp_atol_prop(self) -> XCore.PropertyReal:
        ksp_atol = XCore.PropertyReal(1e-10, 0, 1)
        ksp_atol.Name = "ksp_atol"
        ksp_atol.Description = "KSP Absolute Tolerance"
        return ksp_atol

    # Eigenvalue props

    def _create_num_eigenvals_prop(self) -> XCore.PropertyInt:
        num_eigenvals = XCore.PropertyInt(10, 1)
        num_eigenvals.Name = "num_eigenvals"
        num_eigenvals.Description = "Target Number of Eigenvalues"
        return num_eigenvals

    def _create_target_eigenval_prop(self) -> XCore.PropertyReal:
        target_eigenval = XCore.PropertyReal()
        target_eigenval.Name = "target_eigenval"
        target_eigenval.Description = "Target Eigenvalue"
        target_eigenval.ToolTip = (
            "Solver will try to find the eigenvalues closest in magnitude to this value"
        )
        return target_eigenval

    def _create_eigenval_tol_prop(self) -> XCore.PropertyReal:
        eigenval_tol = XCore.PropertyReal(1e-8, 0, 1)
        eigenval_tol.Name = "eigenval_tol"
        eigenval_tol.Description = "Solver Tolerance"
        return eigenval_tol

    # Time stepping props

    def _create_start_time_prop(self) -> XCore.PropertyReal:
        prop = XCore.PropertyReal(0)
        prop.Name = "start_time"
        prop.Description = "Start Time"
        return prop

    def _create_duration_prop(self) -> XCore.PropertyReal:
        prop = XCore.PropertyReal(10, 0)
        prop.Name = "duration"
        prop.Description = "Duration"
        return prop

    def _create_initial_time_step_size_prop(self) -> XCore.PropertyReal:
        prop = XCore.PropertyReal(1e-3, 0, 1)
        prop.Name = "initial_time_step_size"
        prop.Description = "Initial Time Step Size"
        return prop

    def _create_runge_kutta_method_prop(self) -> XCore.PropertyEnum:
        prop = XCore.PropertyEnum(
            [
                api_models.RungeKuttaMethod.LOBATTO_IIIA.value,
                api_models.RungeKuttaMethod.GAUSS_LEGENDRE.value,
                api_models.RungeKuttaMethod.RADAU_IIA.value,
            ],
            0,
        )
        prop.Name = "runge_kutta_method"
        prop.Description = "Runge Kutta Method"
        return prop

    def _create_num_time_stepping_stages_prop(self) -> XCore.PropertyInt:
        prop = XCore.PropertyInt(3, 1, 10)
        prop.Name = "num_time_stepping_stages"
        prop.Description = "Number of Stages"
        return prop

    def _create_adapt_time_step_prop(self) -> XCore.PropertyBool:
        prop = XCore.PropertyBool(True)
        prop.Name = "adapt_time_step"
        prop.Description = "Adapt Time Step"
        return prop

    def _create_max_time_step_prop(self) -> XCore.PropertyReal:
        prop = XCore.PropertyReal(1e-1, 0, 1000)
        prop.Name = "max_time_step"
        prop.Description = "Max Time Step Size"
        return prop

    def _create_target_error_prop(self) -> XCore.PropertyReal:
        prop = XCore.PropertyReal(1e-3, 0, 1)
        prop.Name = "target_error"
        prop.Description = "Target Error"
        return prop

    def __setstate__(self, state) -> None:
        super().__setstate__(state)

        # backwards compatibility

        # --- v4: Added time stepping solver settings

        if self._time_domain_props.FindChild("initial_time_step_size") is None:
            self._add_time_stepping_props()

        asyncio.get_event_loop().call_soon(
            self._connect_signals
        )  # n.b. ensure de-pickling complete for all objects before trying to connect to other object's signals.

    def _connect_signals(self) -> None:
        solver_type_prop = self._properties.solver_type
        assert isinstance(solver_type_prop, XCore.PropertyEnum)

        solver_type_prop.OnModified.Connect(self._update_props)

        linear_solver_type_prop = self._properties.linear_solver_type
        assert isinstance(linear_solver_type_prop, XCore.PropertyEnum)

        linear_solver_type_prop.OnModified.Connect(self._update_props)

        assert isinstance(self._parent, sim.FenicsSimulation)
        self._parent.simulation_type_prop.OnModified.Connect(self._update_props)

        self._update_props(solver_type_prop, XCore.kPropertyModified)

    def _update_props(
        self, prop: XCore.Property, mod_type: XCore.PropertyModificationTypeEnum
    ) -> None:
        if mod_type != XCore.kPropertyModified:
            return

        # Shared

        nonlin_rtol_prop = self._properties.nonlin_rtol
        assert isinstance(nonlin_rtol_prop, XCore.PropertyReal)

        nonlin_atol_prop = self._properties.nonlin_atol
        assert isinstance(nonlin_atol_prop, XCore.PropertyReal)

        solver_type_prop = self._properties.solver_type
        assert isinstance(solver_type_prop, XCore.PropertyEnum)

        for prop in (nonlin_rtol_prop, nonlin_atol_prop):
            prop.Visible = (
                solver_type_prop.ValueDescription
                == api_models.SolverType.NON_LINEAR.value
            )

        linear_solver_type_prop = self._properties.linear_solver_type
        assert isinstance(linear_solver_type_prop, XCore.PropertyEnum)

        ksp_visible = (
            linear_solver_type_prop.ValueDescription
            == api_models.LinearSolverType.ITERATIVE_KSP.value
        )

        self._ksp_props.Visible = ksp_visible

        # Set all time domain and eigenvalue visible when mode is active

        assert isinstance(self._parent, sim.FenicsSimulation)

        self._eigenvalue_props.Visible = (
            api_models.SimulationType.EIGENVALUE.value
            == self._parent.simulation_type_prop.ValueDescription
        )

        self._time_domain_props.Visible = (
            api_models.SimulationType.TIME_DOMAIN.value
            == self._parent.simulation_type_prop.ValueDescription
        )

    # n.b we do not allow the user to edit the description via the form
    @property
    def description(self) -> str:
        return self._properties.Description

    @description.setter
    def description(self, value: str) -> None:
        self._properties.Description = value

    @property
    def properties(self) -> XCore.PropertyGroup:
        return self._properties

    def validate(self) -> bool:
        solver_type_prop = self._properties.solver_type
        assert isinstance(solver_type_prop, XCore.PropertyEnum)

        num_processes_prop = self._properties.num_processes
        assert isinstance(num_processes_prop, XCore.PropertyInt)

        linear_solver_type_prop = self._properties.linear_solver_type
        assert isinstance(linear_solver_type_prop, XCore.PropertyEnum)

        ksp_pc_prop = self._ksp_props.ksp_pc
        assert isinstance(ksp_pc_prop, XCore.PropertyEnum)

        if (
            solver_type_prop.ValueDescription == api_models.SolverType.LINEAR.value
            and num_processes_prop.Value > 1
            and linear_solver_type_prop.ValueDescription
            == api_models.LinearSolverType.ITERATIVE_KSP.value
            and ksp_pc_prop.ValueDescription
            == api_models.KSPPreConditionerType.ILU.value
        ):
            self.status_icons = [
                "icons/TaskManager/Warning.ico",
            ]
            validation_error = "The ILU preconditioner cannot be used in parallel (more than 1 processor)"
            self.status_icons_tooltip += f"{validation_error}.  "
            logger.error(f"{self.description}: {validation_error}")
            return False

        return True

    def as_api_model(self) -> api_models.SolverSettings:

        # Shared

        field_type_prop = self._properties.field_type
        assert isinstance(field_type_prop, XCore.PropertyEnum)

        num_processes_prop = self._properties.num_processes
        assert isinstance(num_processes_prop, XCore.PropertyInt)

        solver_type_prop = self._properties.solver_type
        assert isinstance(solver_type_prop, XCore.PropertyEnum)

        linear_solver_type_prop = self._properties.linear_solver_type
        assert isinstance(linear_solver_type_prop, XCore.PropertyEnum)

        nonlin_rtol_prop = self._properties.nonlin_rtol
        assert isinstance(nonlin_rtol_prop, XCore.PropertyReal)

        nonlin_atol_prop = self._properties.nonlin_atol
        assert isinstance(nonlin_atol_prop, XCore.PropertyReal)

        # Iterative Solver

        ksp_pc_prop = self._ksp_props.ksp_pc
        assert isinstance(ksp_pc_prop, XCore.PropertyEnum)

        ksp_type_prop = self._ksp_props.ksp_type
        assert isinstance(ksp_type_prop, XCore.PropertyEnum)

        ksp_rtol_prop = self._ksp_props.ksp_rtol
        assert isinstance(ksp_rtol_prop, XCore.PropertyReal)

        ksp_atol_prop = self._ksp_props.ksp_atol
        assert isinstance(ksp_atol_prop, XCore.PropertyReal)

        # Eigenvalue

        num_eigenvals_prop = self._eigenvalue_props.num_eigenvals
        assert isinstance(num_eigenvals_prop, XCore.PropertyInt)

        target_eigenval_prop = self._eigenvalue_props.target_eigenval
        assert isinstance(target_eigenval_prop, XCore.PropertyReal)

        eigenval_tol_prop = self._eigenvalue_props.eigenval_tol
        assert isinstance(eigenval_tol_prop, XCore.PropertyReal)

        # Time Stepping

        start_time_prop = self._time_domain_props.start_time
        assert isinstance(start_time_prop, XCore.PropertyReal)

        duration_prop = self._time_domain_props.duration
        assert isinstance(duration_prop, XCore.PropertyReal)

        runge_kutta_method_prop = self._time_domain_props.runge_kutta_method
        assert isinstance(runge_kutta_method_prop, XCore.PropertyEnum)

        num_time_stepping_stages_prop = self._time_domain_props.num_time_stepping_stages
        assert isinstance(num_time_stepping_stages_prop, XCore.PropertyInt)

        initial_time_step_size_prop = self._time_domain_props.initial_time_step_size
        assert isinstance(initial_time_step_size_prop, XCore.PropertyReal)

        adapt_time_step_prop = self._time_domain_props.adapt_time_step
        assert isinstance(adapt_time_step_prop, XCore.PropertyBool)

        max_time_step_prop = self._time_domain_props.max_time_step
        assert isinstance(max_time_step_prop, XCore.PropertyReal)

        target_error_prop = self._time_domain_props.target_error
        assert isinstance(target_error_prop, XCore.PropertyReal)

        #

        return api_models.SolverSettings(
            field_type=field_type_prop.ValueDescription,
            num_processes=num_processes_prop.Value,
            # Stationary
            solver_type=solver_type_prop.ValueDescription,
            linear_solver_type=linear_solver_type_prop.ValueDescription,
            ksp_pc=ksp_pc_prop.ValueDescription,
            ksp_type=ksp_type_prop.ValueDescription,
            ksp_rtol=ksp_rtol_prop.Value,
            ksp_atol=ksp_atol_prop.Value,
            nonlin_rtol=nonlin_rtol_prop.Value,
            nonlin_atol=nonlin_atol_prop.Value,
            # Eigenvalue
            num_eigenvals=num_eigenvals_prop.Value,
            target_eigenval=target_eigenval_prop.Value,
            eigenval_tol=eigenval_tol_prop.Value,
            # Time Stepping
            start_time=start_time_prop.Value,
            duration=duration_prop.Value,
            runge_kutta_method=runge_kutta_method_prop.ValueDescription,
            num_time_stepping_stages=num_time_stepping_stages_prop.Value,
            initial_time_step_size=initial_time_step_size_prop.Value,
            adapt_time_step=adapt_time_step_prop.Value,
            max_time_step=max_time_step_prop.Value,
            target_error=target_error_prop.Value,
        )
