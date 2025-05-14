import asyncio
import logging
from abc import abstractmethod
from itertools import chain
from typing import TYPE_CHECKING

import fenics.model.fenics_simulation as sim  # Simulation, domain_id_map_t, logger
import XCore as xc
import XCoreHeadless
import XCoreModeling as xm
from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem
from s4l_core.simulator_plugins.base.model.group import Group
from s4l_core.simulator_plugins.base.model.help import create_help_button
from fenics.model.utils import is_valid_variable_name
from fenics.solver.driver import api_models
from XCorePython import PropertyExpression

if TYPE_CHECKING:
    from fenics.model.settings import (
        AllBoundaryFluxSettings,
        AllDirichletSettings,
        AllSubdomainSettings,
    )

logger = logging.getLogger(__name__)


class Equation(TreeItem):
    def __init__(
        self,
        parent: TreeItem,
        subdomain_settings: "AllSubdomainSettings",
        boundary_flux_settings: "AllBoundaryFluxSettings",
        dirichlet_settings: "AllDirichletSettings",
        description_: str = "Equation",
        variable_shape: str = api_models.VariableType.SCALAR.value,
        icon: str = "icons/XPostProcessor/PhysicalScalarQuantity.ico",
        variable_name: str = "u",
    ) -> None:
        super().__init__(parent, icon=icon)

        description = xc.PropertyString(description_)
        description.Description = "Description"

        variable = PropertyExpression(variable_name)
        variable.Description = "Variable Name"

        element_type = xc.PropertyEnum(
            [
                "Lagrange",
                "Discontinuous Lagrange",
                "Crouzeix-Raviart",
                "Brezzi-Douglas-Marini",
                "Brezzi-Douglas-Fortin-Marini",
                "Raviart-Thomas",
                "Nedelec 1st kind H(div)",
                "Nedelec 2st kind H(div)",
                "Nedelec 1st kind H(curl)",
                "Nedelec 2st kind H(curl)",
            ],
            0,
        )
        element_type.Description = "Element Type"

        element_degree = xc.PropertyInt(1, 0, 3)
        element_degree.Description = "Element Degree"
        element_degree.ToolTip = (
            "e.g. 1 implies linear interpolation between the element nodes"
        )

        self._properties: XCoreHeadless.DialogOptions = XCoreHeadless.DialogOptions()
        self._properties.Add("help_button", create_help_button())
        self._properties.Add("description", description)
        self._properties.Add("variable", variable)

        self._properties.Add("element_type", element_type)
        self._properties.Add("element_degree", element_degree)

        initial_condition = PropertyExpression("0")
        initial_condition.Description = "Initial Condition"
        self._properties.Add("initial_condition", initial_condition)

        #

        self._unstructured_mesh_id: str | None = None

        self._subdomain_settings = subdomain_settings
        self._boundary_flux_settings = boundary_flux_settings
        self._dirichlet_settings = dirichlet_settings

        self._variable_shape = variable_shape

        #

        asyncio.get_event_loop().call_soon(
            self._connect_signals
        )  # n.b. ensure de-pickling complete for all objects before trying to connect to other object's signals.

    @property
    def variable_shape(self) -> str:
        return self._variable_shape

    @abstractmethod
    def _display_help(self) -> None:
        """
        Calculate the help message and then display it.

        Should use display_help from the fenics_plugin.model.help module to display the help message
        """
        ...

    def __setstate__(self, state):
        super().__setstate__(state)
        asyncio.get_event_loop().call_soon(
            self._connect_signals
        )  # n.b. ensure de-pickling complete for all objects before trying to connect to other object's signals.

    def _connect_signals(self) -> None:
        help_button = self._properties.help_button
        assert isinstance(help_button, xc.PropertyPushButton)
        help_button.OnClicked.Connect(self._display_help)

        def description_changed(
            prop: xc.Property, mod_type: xc.PropertyModificationTypeEnum
        ):
            if mod_type != xc.kPropertyModified:
                return

            self._notify_modified(False)

        for prop in (self._properties.description, self._properties.variable):
            assert isinstance(prop, xc.Property)
            prop.OnModified.Connect(description_changed)

    @property
    def parent_sim_type_prop(self) -> xc.PropertyEnum:
        parent_sim = self.parent.parent
        assert isinstance(parent_sim, sim.FenicsSimulation)
        return parent_sim.simulation_type_prop

    @property
    def unstructured_mesh(self) -> xm.UnstructuredMesh | None:
        if self._unstructured_mesh_id is None:
            return None

        entity = xm.GetActiveModel().LookupEntity(xc.Uuid(self._unstructured_mesh_id))
        assert isinstance(entity, xm.UnstructuredMesh)
        return entity

    @property
    def variable_name(self) -> str:
        prop = self._properties.variable
        assert isinstance(prop, PropertyExpression)
        return prop.Value

    @variable_name.setter
    def variable_name(self, name: str) -> None:
        self._properties.variable.Value = name

    @property
    def description(self) -> str:
        return f"{self._properties.description.Value} - {self.variable_name}"

    @description.setter
    def description(self, value: str) -> None:
        self._properties.description.Value = value

    @property
    def unstructured_mesh_id(self) -> str | None:
        return self._unstructured_mesh_id

    @property
    def properties(self) -> XCoreHeadless.DialogOptions:
        return self._properties

    #

    @property
    def subdomain_settings(self) -> "AllSubdomainSettings":
        return self._subdomain_settings

    @property
    def boundary_flux_settings(self) -> "AllBoundaryFluxSettings":
        return self._boundary_flux_settings

    @property
    def dirichlet_settings(self) -> "AllDirichletSettings":
        return self._dirichlet_settings

    #

    def clear_status_recursively(self):
        self.clear_status()
        for group in (
            self._subdomain_settings,
            self._boundary_flux_settings,
            self._dirichlet_settings,
        ):
            group.clear_status_recursively()

    def validate(self) -> bool:
        self.clear_status_recursively()

        if not is_valid_variable_name(self.variable_name):
            self.status_icons = [
                "icons/TaskManager/Warning.ico",
            ]
            validation_error = (
                f"Variable Name '{self.variable_name}' is not allowed, please"
            )
            self.status_icons_tooltip += f"{validation_error}.  "
            logger.error(f"{self.description}: {validation_error}")

        self._unstructured_mesh_id = None
        unstructured_mesh = None

        # Check mesh entities are (still) valid

        ok = True
        for sub_settings in chain(
            self._subdomain_settings.elements,
            self._dirichlet_settings.elements,
            self._boundary_flux_settings.elements,
        ):
            for geometry in sub_settings.geometries:
                entity = xm.GetActiveModel().LookupEntity(xc.Uuid(geometry.entity_id))
                if entity is None:
                    geometry.status_icons = [
                        "icons/TaskManager/Warning.ico",
                    ]
                    validation_error = "This geometry no longer exists - perhaps the parent mesh was deleted"
                    geometry.status_icons_tooltip += f"{validation_error}.  "
                    logger.error(f"{geometry.description}: {validation_error}")

                    ok = False
                    continue

        if not ok:
            return False

        # Detect unstructured mesh

        for sub_settings in self._subdomain_settings.elements:
            if len(sub_settings.geometries) == 0:
                continue

            ent = sub_settings.geometries[0].entity
            assert ent is not None  # checked above

            unstructured_mesh = ent.Mesh
            self._unstructured_mesh_id = unstructured_mesh.Id.str()
            break

        if self._unstructured_mesh_id is None:
            self.status_icons = [
                "icons/TaskManager/Warning.ico",
            ]
            validation_error = "No mesh domains assigned to equation"
            self.status_icons_tooltip += f"{validation_error}.  "
            logger.error(f"{self.description}: {validation_error}")
            return False

        assert unstructured_mesh is not None

        # Check assigned geometries are children of parent mesh

        ok = True
        for sub_settings in chain(
            self._subdomain_settings.elements,
            self._dirichlet_settings.elements,
            self._boundary_flux_settings.elements,
        ):
            for geometry in sub_settings.geometries:
                ent = geometry.entity
                assert ent is not None

                if ent.Mesh.Id.str() != self._unstructured_mesh_id:
                    geometry.status_icons = [
                        "icons/TaskManager/Warning.ico",
                    ]
                    validation_error = f"This geometry does not come from the expected mesh: '{unstructured_mesh.Name}'"
                    geometry.status_icons_tooltip += f"{validation_error}.  "
                    logger.error(f"{geometry.description}: {validation_error}")

                    ok = False
                    continue

        # Check subdomains appear exactly once in settings
        domain_settings: dict[str, list[str]] = {
            domain.Id.str(): [] for domain in unstructured_mesh.Domains
        }  # map domain_id to settings description

        for sub_settings in self._subdomain_settings.elements:
            for geometry in sub_settings.geometries:
                domain_id = geometry.entity_id
                assert domain_id in domain_settings

                if len(domain_settings[domain_id]) > 0:
                    geometry.status_icons = [
                        "icons/TaskManager/Warning.ico",
                    ]
                    validation_error = f"This domain is already associated with settings: {domain_settings[domain_id][0]}"
                    geometry.status_icons_tooltip += f"{validation_error}.  "
                    logger.error(f"{geometry.description}: {validation_error}")
                    ok = False

                else:
                    domain_settings[domain_id].append(sub_settings.description)

        subdomains_without_settings: list[str] = []
        for domain_id, settings in domain_settings.items():
            if len(settings) == 0:
                entity = xm.GetActiveModel().LookupEntity(xc.Uuid(domain_id))
                assert isinstance(entity, xm.UnstructuredMeshDomain)
                subdomains_without_settings.append(entity.Name)

        if len(subdomains_without_settings) > 0:
            self._subdomain_settings.status_icons = [
                "icons/TaskManager/Warning.ico",
            ]
            validation_error = f"The following subdomains lack settings: {', '.join(subdomains_without_settings)}"
            self._subdomain_settings.status_icons_tooltip += f"{validation_error}.  "
            logger.error(f"{self._subdomain_settings.description}: {validation_error}")
            ok = False

        # check patches only appear once in dirichlet and flux b.c.
        patch_settings: dict[str, list[str]] = {
            patch.Id.str(): [] for patch in unstructured_mesh.Patches
        }

        ok = True
        for sub_settings in chain(
            self._boundary_flux_settings.elements,
            self._dirichlet_settings.elements,
        ):
            for geometry in sub_settings.geometries:
                patch_id = geometry.entity_id

                if len(patch_settings[patch_id]) > 0:
                    geometry.status_icons = [
                        "icons/TaskManager/Warning.ico",
                    ]
                    validation_error = f"This patch is already associated with settings: {patch_settings[patch_id][0]}"
                    geometry.status_icons_tooltip += f"{validation_error}.  "
                    logger.error(f"{geometry.description}: {validation_error}")
                    ok = False

                elif patch_id in patch_settings:
                    patch_settings[patch_id].append(sub_settings.description)

                else:
                    pass  # is not a child of parent mesh

        return ok

    def as_api_model(self) -> api_models.Equation:
        if not self.validate():
            raise RuntimeError("Failed to validate aborting serialization")

        assert self.unstructured_mesh is not None

        #

        element_type_prop = self._properties.element_type
        assert isinstance(element_type_prop, xc.PropertyEnum)

        element_degree_prop = self._properties.element_degree
        assert isinstance(element_degree_prop, xc.PropertyInt)

        #

        domain_id_map: "sim.domain_id_map_t" = {}

        for domain in self.unstructured_mesh.Domains:
            assert isinstance(domain, xm.UnstructuredMeshDomain)
            domain_id_map[str(domain.Id)] = domain.Index

        patch_id_map: "sim.domain_id_map_t" = {}

        for idx, patch in enumerate(self.unstructured_mesh.Patches):
            assert isinstance(patch, xm.UnstructuredMeshPatch)
            patch_id_map[str(patch.Id)] = idx + 1

        #
        variable_prop = self._properties.variable
        assert isinstance(variable_prop, PropertyExpression)
        variable_name = variable_prop.Value

        #

        ic_prop = self._properties.initial_condition
        assert isinstance(ic_prop, PropertyExpression)
        initial_condition = ic_prop.Value

        #

        weak_forms = []

        for domain_setting in self._subdomain_settings.elements:
            if (
                weak_form := domain_setting.as_api_model(variable_name, domain_id_map)
            ) is not None:
                weak_forms.append(weak_form)

        for boundary_flux_setting in self._boundary_flux_settings.elements:
            if (
                weak_form := boundary_flux_setting.as_api_model(
                    variable_name, patch_id_map
                )
            ) is not None:
                weak_forms.append(weak_form)

        #

        dirichlet_conditions = []

        for dirichlet_setting in self._dirichlet_settings.elements:
            if (
                dirichlet_condition := dirichlet_setting.as_api_model(
                    variable_name, patch_id_map
                )
            ) is not None:
                dirichlet_conditions.append(dirichlet_condition)

        return api_models.Equation(
            variable_name=variable_name,
            variable_shape=self._variable_shape,
            element_type=element_type_prop.ValueDescription,
            element_degree=element_degree_prop.Value,
            weak_form=weak_forms,
            dirichlet_conditions=dirichlet_conditions,
            initial_condition=initial_condition,
        )


class Equations(Group[Equation]):
    def __init__(
        self, parent: TreeItem, is_expanded: bool = True, icon: str = ""
    ) -> None:
        super().__init__(
            parent, Equation, is_expanded, icon="icons/XPostProcessor/gradient.ico"
        )

    def _get_new_element_description(self) -> str:
        return "Equation"

    def clear_status_recursively(self):
        self.clear_status()
        for eq in self._elements:
            eq.clear_status_recursively()

    @property
    def description_text(self) -> str:
        return f"{self._get_new_element_description()}s"
