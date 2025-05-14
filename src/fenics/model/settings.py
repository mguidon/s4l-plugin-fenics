import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import XCore as xc
import XCoreHeadless
import XCoreModeling as xm
from s4l_core.simulator_plugins.base.model.group import Group
from s4l_core.simulator_plugins.base.model.help import create_help_button, display_help
from fenics.model.equation_settings import (
    EquationSettings,
)
from fenics.solver.driver import api_models

if TYPE_CHECKING:
    from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem


subdomain_settings_t = EquationSettings[xm.UnstructuredMeshDomain, api_models.WeakTerm]
dirichlet_settings_t = EquationSettings[
    xm.UnstructuredMeshPatch, api_models.DirichletCondition
]
boundary_flux_settings_t = EquationSettings[
    xm.UnstructuredMeshPatch, api_models.WeakTerm
]

SettingsT = TypeVar(
    "SettingsT", subdomain_settings_t, boundary_flux_settings_t, dirichlet_settings_t
)


class AllSettingsBase(Group[SettingsT], Generic[SettingsT]):
    def __init__(
        self,
        parent: "TreeItem",
        settings_t: type[SettingsT],
        new_element_description: str,
        icon: str,
    ) -> None:
        super().__init__(
            parent,
            settings_t,
            is_expanded=True,
            icon=icon,
        )

        self._new_element_description = new_element_description

        self._properties = XCoreHeadless.DialogOptions()
        self._properties.Add("help_button", create_help_button())

        asyncio.get_event_loop().call_soon(self._connect_signals)

    @property
    def properties(self) -> xc.PropertyGroup:
        return self._properties

    def __setstate__(self, state):
        super().__setstate__(state)
        asyncio.get_event_loop().call_soon(
            self._connect_signals
        )  # n.b. ensure de-pickling complete for all objects before trying to connect to other object's signals.

    def _connect_signals(self) -> None:
        help_button = self._properties.help_button
        assert isinstance(help_button, xc.PropertyPushButton)
        help_button.OnClicked.Connect(self._display_help)

    @abstractmethod
    def _display_help(self) -> None:
        """
        Calculate the help message and then display it.

        Should use display_help from the fenics_plugin.model.help module to display the help message
        """
        ...

    def _get_new_element_description(self) -> str:
        return self._new_element_description

    def clear_status_recursively(self) -> None:
        self.clear_status()
        for element in self._elements:
            element.clear_status_recursively()

    @property
    def description_text(self) -> str:
        return f"All {self._get_new_element_description()}"


class AllSubdomainSettings(AllSettingsBase[subdomain_settings_t]):
    def __init__(
        self,
        parent: "TreeItem",
        settings_t: type[subdomain_settings_t],
        new_element_description: str,
    ) -> None:
        super().__init__(
            parent,
            settings_t,
            new_element_description=new_element_description,
            icon="icons/XMaterials/material.ico",
        )

    def _display_help(self) -> None:
        text = """
This folder contains all sets of subdomain settings.

Drag and drop a subdomain here to create a new subdomain settings for it.

All subdomains in the parent mesh need to be assigned to a settings.

See the help of an individual subdomain settings for more information.
"""
        display_help("All Subdomain Settings", text)


class AllBoundaryFluxSettings(AllSettingsBase[boundary_flux_settings_t]):
    def __init__(
        self,
        parent: "TreeItem",
        settings_t: type[boundary_flux_settings_t],
        new_element_description: str,
    ) -> None:
        super().__init__(
            parent,
            settings_t,
            new_element_description=new_element_description,
            icon="icons/FieldViewers/ExtractSurface.ico",
        )

    def _display_help(self) -> None:
        text = """
This folder contains all sets of boundary flux settings.

Drag and drop a patch (surface) here to create a new settings for it.

See the help of an individual boundary flux settings for more information.

Note that by default external surfaces to which no dirichlet or flux boundary condition is applied
will default to the zero flux boundary condition.
"""
        display_help("All Subdomain Settings", text)


class AllDirichletSettings(AllSettingsBase[dirichlet_settings_t]):
    def __init__(
        self,
        parent: "TreeItem",
        settings_t: type[dirichlet_settings_t],
        new_element_description: str,
    ) -> None:
        super().__init__(
            parent,
            settings_t,
            new_element_description=new_element_description,
            icon="icons/XModelerUI/SurfaceProjection.ico",
        )

    def _display_help(self) -> None:
        text = """
This folder contains all sets of dirichlet boundary settings.

Drag and drop a patch (surface) here to create a new settings for it.

See the help of an individual dirichlet boundary settings for more information.
"""
        display_help("All Subdomain Settings", text)
