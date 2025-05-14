import logging
from typing import TYPE_CHECKING, Any, Optional, cast

import s4l_core.simulator_plugins.base.model.geometry as geometry
import fenics.model.equation as equation
import fenics.model.equation_settings as equation_settings
import fenics.model.expressions as expressions
import fenics.model.fenics_simulation as fenics_sim
import fenics.model.pde as pde
import fenics.model.settings as settings
import fenics.model.solid_mechanics as solid_mech
import fenics.model.solver_settings as solver_settings
import XController
import XCoreHeadless
from s4l_core.simulator_plugins.base.controller.simulation_manager_interface import (
    ISimulationManager,
)
from fenics.model.weak_form import WeakFormEquation

if TYPE_CHECKING:
    from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem

logger = logging.getLogger(__name__)


class FenicsSimulationManager(ISimulationManager):
    """Manager implementation for FenicsSimulation."""

    def __init__(self, simulation: fenics_sim.FenicsSimulation):
        super().__init__(simulation)

        self._new_pde_action = XController.Action(
            "PDE",
            "Add an equation of type PDE",
            XController.Icon("icons/XPostProcessor/PhysicalScalarQuantity.ico"),
        )
        self._new_pde_action.OnTriggered.Connect(self.on_new_pde_triggered)

        self._new_solid_mech_eq_action = XController.Action(
            "Solid Mechanics",
            "Add an equation of solid mechanics",
            XController.Icon("icons/XModelerUI/create_solidblock.ico"),
        )
        self._new_solid_mech_eq_action.OnTriggered.Connect(
            self.on_new_solid_mech_eq_triggered
        )

        self._new_weak_form_eq_action = XController.Action(
            "Weak Form",
            "Define the weak form directly",
            XController.Icon("icons/MeshingUI/mesh.ico"),
        )
        self._new_weak_form_eq_action.OnTriggered.Connect(
            self.on_new_weak_form_eq_triggered
        )

        self._new_subdomain_settings_action = XController.Action(
            "Subdomain",
            "Add a new set of subdomain settings to the current equation",
            XController.Icon("icons/XMaterials/material.ico"),
        )
        self._new_subdomain_settings_action.OnTriggered.Connect(
            self.on_new_subdomain_settings_triggered
        )

        self._new_boundary_flux_settings_action = XController.Action(
            "Flux Settings",
            "Add a new set of flux settings to the current equation",
            XController.Icon("icons/FieldViewers/ExtractSurface.ico"),
        )
        self._new_boundary_flux_settings_action.OnTriggered.Connect(
            self.on_new_boundary_flux_settings_triggered
        )

        self._new_dirichlet_settings_action = XController.Action(
            "Dirichlet Settings",
            "Add a new set of flux settings to the current equation",
            XController.Icon("icons/XModelerUI/SurfaceProjection.ico"),
        )
        self._new_dirichlet_settings_action.OnTriggered.Connect(
            self.on_new_dirichlet_settings_triggered
        )

    @property
    def simulation(self) -> fenics_sim.FenicsSimulation:
        """Get the simulation with the correct type."""
        return cast("fenics_sim.FenicsSimulation", self._simulation)

    def collect_actions(self, selection: Any) -> list[XController.Action]:
        """
        Collect FenicsSimulation-specific actions.

        Returns:
            List of actions specific to FenicsSimulation
        """

        self._new_pde_action.Enabled = (
            len(selection) == 1 and len(selection[0].Path()) > 1
        )
        self._new_solid_mech_eq_action.Enabled = (
            len(selection) == 1 and len(selection[0].Path()) > 1
        )
        self._new_weak_form_eq_action.Enabled = (
            len(selection) == 1 and len(selection[0].Path()) > 1
        )
        self._new_subdomain_settings_action.Enabled = (
            len(selection) == 1 and len(selection[0].Path()) > 3
        )
        self._new_boundary_flux_settings_action.Enabled = (
            len(selection) == 1 and len(selection[0].Path()) > 3
        )
        self._new_dirichlet_settings_action.Enabled = (
            len(selection) == 1 and len(selection[0].Path()) > 3
        )
        return [
            self._new_pde_action,
            self._new_solid_mech_eq_action,
            self._new_weak_form_eq_action,
            self._new_subdomain_settings_action,
            self._new_boundary_flux_settings_action,
            self._new_dirichlet_settings_action,
        ]

    def update_properties(
        self,
        properties_registry: XCoreHeadless.PropertyRegistry,
        selected_item: "TreeItem",
        parent_item: Optional["TreeItem"],
    ) -> None:
        """
        Update properties for MySimulation items.
        """
        # Handle Settings
        if isinstance(
            selected_item,
            (
                settings.AllSettingsBase,
                equation_settings.EquationSettings,
                equation.Equation,
                solver_settings.SolverSettings,
                expressions.Expressions,
                fenics_sim.FenicsSimulation,
            ),
        ):
            properties_registry.SetProperties([selected_item.properties])
            return

        # Handle Geometries -> Show properties of the parent item
        if isinstance(selected_item, geometry.Geometry):
            if isinstance(
                parent_item,
                (equation_settings.EquationSettings,),
            ):
                properties_registry.SetProperties([parent_item.properties])
                return

        # For any other items, clear properties
        properties_registry.SetProperties([])

    def _on_new_equation(self, eq_class: type[equation.Equation]) -> None:
        self.simulation.equations.add(eq_class)

    def on_new_pde_triggered(self) -> None:
        return self._on_new_equation(pde.PDE)

    def on_new_solid_mech_eq_triggered(self) -> None:
        return self._on_new_equation(solid_mech.LinearElasticityEquation)

    def on_new_weak_form_eq_triggered(self) -> None:
        return self._on_new_equation(WeakFormEquation)

    def on_new_subdomain_settings_triggered(self) -> None:
        assert self._tree is not None
        selection = self._tree.SelectedNodes()
        assert len(selection) == 1
        path: list[int] = selection[0].Path()
        assert len(path) > 3

        equation = self.simulation.equations.elements[path[3]]
        equation.subdomain_settings.add()

    def on_new_boundary_flux_settings_triggered(self) -> None:
        assert self._tree is not None
        selection = self._tree.SelectedNodes()
        assert len(selection) == 1
        path: list[int] = selection[0].Path()
        assert len(path) > 3

        equation = self.simulation.equations.elements[path[3]]
        equation.boundary_flux_settings.add()

    def on_new_dirichlet_settings_triggered(self) -> None:
        assert self._tree is not None
        selection = self._tree.SelectedNodes()
        assert len(selection) == 1
        path: list[int] = selection[0].Path()
        assert len(path) > 3

        equation = self.simulation.equations.elements[path[3]]
        equation.dirichlet_settings.add()
