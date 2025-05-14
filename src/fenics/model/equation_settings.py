import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import XCore
import XCore as xc
import XCoreModeling as xm
from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem
from s4l_core.simulator_plugins.base.model.geometry_interface import HasGeometries
from s4l_core.simulator_plugins.base.model.help import create_help_button
from fenics.model.equation import Equation
from XCoreHeadless import DialogOptions

if TYPE_CHECKING:
    from fenics.model.fenics_simulation import (
        domain_id_map_t,
    )


EntityT = TypeVar("EntityT", bound=xm.Entity)
ApiMdlT = TypeVar("ApiMdlT")


class EquationSettings(HasGeometries, Generic[EntityT, ApiMdlT]):
    def __init__(
        self,
        parent: TreeItem,
        description: str = "Settings",
        icon: str = "icons/MashEnvironment/Folder.ico",
    ) -> None:
        # Get allowed types before initializing HasGeometries
        allowed_types = self._get_allowed_entity_types()

        super().__init__(
            parent=parent,
            is_expanded=False,
            icon=icon,
            allowed_entity_types=allowed_types,
        )

        self._properties = DialogOptions()

        self._properties.Add("help_button", create_help_button())

        description_prop = xc.PropertyString(description)
        self._properties.Add("description", description_prop)
        description_prop.Description = "Description"

        asyncio.get_event_loop().call_soon(
            self._connect_signals
        )  # schedule this for after the child has had a chance to finish building the props in its __init__

    def _get_allowed_entity_types(self) -> tuple[type[xm.Entity], ...]:
        """
        Get the allowed entity types based on the class's entity type parameter.

        This uses the Generic EntityT parameter to determine which entity types are allowed.

        Returns:
            Tuple of allowed entity types
        """
        return (self._get_cell_cls(),)

    @abstractmethod
    def _display_help(self) -> None:
        """
        Calculate the help message and then display it.

        Should use display_help from the fenics_plugin.model.help module to display the help message
        """
        ...

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        asyncio.get_event_loop().call_soon(
            self._connect_signals
        )  # n.b. ensure de-pickling complete for all objects before trying to connect to other object's signals.

    def _connect_signals(self) -> None:
        help_button = self._properties.help_button
        assert isinstance(help_button, xc.PropertyPushButton)
        help_button.OnClicked.Connect(self._display_help)

        def description_changed(
            prop: XCore.Property, mod_type: XCore.PropertyModificationTypeEnum
        ):
            if mod_type != XCore.kPropertyModified:
                return

            self._notify_modified(False)

        prop = self._properties.description
        assert isinstance(prop, XCore.Property)
        prop.OnModified.Connect(description_changed)

    @property
    def description(self) -> str:
        return f"{self._properties.description.Value} ({len(self.geometries)})"

    @description.setter
    def description(self, value: str) -> None:
        self._properties.description.Value = value

    @property
    def properties(self) -> xc.PropertyGroup:
        return self._properties

    @property
    def parent_equation(self) -> "Equation":
        parent_eq = self.parent.parent
        assert isinstance(parent_eq, Equation)
        return parent_eq

    @abstractmethod
    def _get_cell_cls(self) -> type[xm.Entity]:
        # get_orig_class from pytypes not working with inheritance?
        ...

    @property
    def _parent_sim_type_prop(self) -> XCore.PropertyEnum:
        return self.parent_equation.parent_sim_type_prop

    def clear_status_recursively(self):
        self.clear_status()
        for geometry in self._geometries:
            geometry.clear_status()

    def _domain_ids(self, domain_id_map: "domain_id_map_t") -> list[int]:
        domain_ids = []
        for geometry in self._geometries:
            if str(geometry.entity_id) not in domain_id_map:
                raise ValueError(
                    f"Could not find ent: {geometry.description} in domain_id_map"
                )

            domain_ids.append(domain_id_map[str(geometry.entity_id)])

        return domain_ids

    def _domain_ids_as_str(self, domain_id_map: "domain_id_map_t") -> str:
        return ",".join(
            [str(domain_id) for domain_id in self._domain_ids(domain_id_map)]
        )

    @abstractmethod
    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> ApiMdlT | None:
        """
        Return the api_model instance or None if the settings has no
        assigned geometry
        """
        ...
