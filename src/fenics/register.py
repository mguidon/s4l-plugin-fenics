import logging
from typing import cast

from s4l_core.simulator_plugins.base.controller.simulation_binding_interface import (
    ISimulationBinding,
)
from s4l_core.simulator_plugins.base.controller.simulation_manager_interface import (
    ISimulationManager,
)
from s4l_core.simulator_plugins.base.model.simulation_base import SimulationBase
from s4l_core.simulator_plugins.common.registry import PluginRegistry
from fenics.controller.fenics_simulation_binding import (
    FenicsSimulationBinding,
)
from fenics.controller.fenics_simulation_manager import (
    FenicsSimulationManager,
)
from fenics.model.fenics_simulation import (
    FenicsSimulation,
)

logger = logging.getLogger(__name__)


def create_fenics_binding(simulation: SimulationBase) -> ISimulationBinding:
    """Factory function to create a binding for FenicsSimulation."""
    return FenicsSimulationBinding(cast(FenicsSimulation, simulation))


def create_fenics_manager(simulation: SimulationBase) -> ISimulationManager:
    """Factory function to create a manager for FenicsSimulation."""
    return FenicsSimulationManager(cast(FenicsSimulation, simulation))


def register():
    """Register FenicsSimulation components."""
    logger.info("Registering FenicsSimulation...")

    # Register simulation class
    PluginRegistry.register_simulation(FenicsSimulation)

    # Register binding factory
    sim_type = FenicsSimulation.get_simulation_type_name()
    PluginRegistry.register_binding_factory(sim_type, create_fenics_binding)

    # Register manager factory
    PluginRegistry.register_manager_factory(sim_type, create_fenics_manager)

    logger.info("Registered FenicsSimulation components successfully")
