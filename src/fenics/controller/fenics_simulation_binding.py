import logging
from typing import TYPE_CHECKING, Optional, cast

import fenics.model.fenics_simulation as my_sim
from s4l_core.simulator_plugins.base.controller.simulation_binding_interface import (
    ISimulationBinding,
)

if TYPE_CHECKING:
    from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem

logger = logging.getLogger(__name__)


class FenicsSimulationBinding(ISimulationBinding):
    """Binding implementation for FenicsSimulation."""

    @property
    def simulation(self) -> my_sim.FenicsSimulation:
        """Get the simulation instance with the correct type."""
        return cast("my_sim.FenicsSimulation", self._simulation)

    def count_children(self, path: list[int]) -> int:
        """
        Count implementation for MySimulation.

        Args:
            path: The tree path

        Returns:
            Number of children
        """
        simulation = self.simulation

        if (
            len(path) == 2
        ):  # solver settings, global expressions, equations, postpro expressions
            return 4

        if len(path) == 3:
            if path[2] in (0, 1, 3):  # solver settings, global expressions
                return 0  # these settings have no children
            elif path[2] == 2:
                return len(simulation.equations.elements)
            else:
                raise RuntimeError("Invalid path")

        equation = simulation.equations.elements[int(path[3])]

        if len(path) == 4:  # an equation
            assert path[2] == 2  # equations node
            return 3  # subdomain settings, flux settings, dirichlet settings

        settings_type_idx = int(path[4])

        if settings_type_idx == 0:
            settings_group = equation.subdomain_settings
        elif settings_type_idx == 1:
            settings_group = equation.boundary_flux_settings
        elif settings_type_idx == 2:
            settings_group = equation.dirichlet_settings
        else:
            raise RuntimeError("Invalid path")

        if len(path) == 5:  # an equation settings
            return len(settings_group.elements)

        settings = settings_group.elements[int(path[5])]

        if len(path) == 6:
            return len(settings.geometries)

        return 0

    def get_tree_item(self, path: list[int]) -> Optional["TreeItem"]:
        """
        Get tree item implementation for FenicsSimulation.

        Args:
            path: The tree path

        Returns:
            The tree item at the path or None
        """
        simulation = self.simulation

        if len(path) == 2:
            return simulation

        if path[2] == 0:
            sim_child = simulation.solver_settings
        elif path[2] == 1:
            sim_child = simulation.global_expressions
        elif path[2] == 2:
            sim_child = simulation.equations
        elif path[2] == 3:
            sim_child = simulation.post_processing_expressions
        else:
            logger.error(f"Invalid index for simulation child: {path[2]}")
            return None

        if len(path) == 3:
            return sim_child

        if path[3] >= len(simulation.equations.elements):
            logger.debug("Out of range request for equation")
            return None

        equation = simulation.equations.elements[path[3]]

        if len(path) == 4:
            assert path[2] == 2
            return equation

        settings_type_idx = path[4]

        if settings_type_idx == 0:
            settings_group = equation.subdomain_settings
        elif settings_type_idx == 1:
            settings_group = equation.boundary_flux_settings
        elif settings_type_idx == 2:
            settings_group = equation.dirichlet_settings
        else:
            logger.error(f"Invalid index for equation child: {path[2]}")
            return None

        if len(path) == 5:
            return settings_group

        if path[5] >= len(settings_group.elements):
            logger.debug("Out of range request for settings")
            return None

        settings = settings_group.elements[path[5]]

        if len(path) == 6:
            return settings

        if path[6] >= len(settings.geometries):
            logger.debug("Out of range request for geometry")
            return None

        return settings.geometries[path[6]]
