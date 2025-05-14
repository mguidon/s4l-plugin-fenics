import logging
import uuid
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Generic

import XCore
import XCoreHeadless
import XCoreModeling as xm
from s4l_core.simulator_plugins.base.model.help import display_help
from fenics.model.equation import Equation
from fenics.model.equation_settings import (
    EntityT,
    EquationSettings,
)
from fenics.model.expressions import EXPRESSION_INFO
from fenics.model.settings import (
    AllBoundaryFluxSettings,
    AllDirichletSettings,
    AllSubdomainSettings,
)
from fenics.model.utils import is_valid_variable_name
from fenics.solver.driver import api_models as mdl
from XCore import PropertyEnum
from XCorePython import PropertyExpression

if TYPE_CHECKING:
    from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem
    from fenics.model.fenics_simulation import (
        domain_id_map_t,
    )


logger = logging.getLogger(__name__)


class WeakTermType(Enum):
    LINEAR = "Linear: L(v)"
    BILINEAR = "Bi-Linear: a(u,v)"


TRIAL_TEST_FUNCTION_EXPLANATION = """
- The trial function is equal to the equation's variable name
- The test function is the equation's variable name with the suffix: _test
"""


class WeakFormEquationSettings(
    EquationSettings[EntityT, mdl.WeakTerm], Generic[EntityT]
):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(parent)

        add_term_prop = XCore.PropertyPushButton("+")
        self._properties.Add("add", add_term_prop)

    def _connect_signals(self) -> None:
        super()._connect_signals()

        add_expression_prop = self._properties.add
        assert isinstance(add_expression_prop, XCore.PropertyPushButton)

        add_expression_prop.OnClicked.Connect(self.add)

        for child_group in self._properties.Children:
            if not isinstance(child_group, XCoreHeadless.DialogOptions):
                continue

            expr_delete = child_group.delete
            assert isinstance(expr_delete, XCore.PropertyPushButton)

            def remove_term():
                child_group.Lose()

            expr_delete.OnClicked.Connect(remove_term)

    def add(self, value_: str = "", type_: int = 0) -> None:
        term_type = PropertyEnum(
            (WeakTermType.LINEAR.value, WeakTermType.BILINEAR.value), type_
        )

        term_value = PropertyExpression(value_)
        term_delete = XCore.PropertyPushButton("Delete")
        term_delete.Icon = "icons/XCoreUI/Delete.ico"

        term_group = XCoreHeadless.DialogOptions()
        term_group.Description = " "

        term_group.Add("type", term_type)
        term_group.Add("value", term_value)
        term_group.Add("delete", term_delete)

        self._properties.Add(str(uuid.uuid4()), term_group)

        def remove_term():
            term_group.Lose()

        term_delete.OnClicked.Connect(remove_term)

    @abstractmethod
    def _integral_domain_as_str(self, domain_id_map: "domain_id_map_t") -> str:
        ...

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> mdl.WeakTerm | None:
        if not is_valid_variable_name(variable_name):
            raise ValueError(
                f"Tried to form api_mdl description with invalid variable name: {variable_name}"
            )

        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        L_terms = []
        a_terms = []

        domain = self._integral_domain_as_str(domain_id_map)

        for child_group in self._properties.Children:
            if not isinstance(child_group, XCoreHeadless.DialogOptions):
                continue

            term_type_prop = child_group.type
            assert isinstance(term_type_prop, XCore.PropertyEnum)

            term_expr_prop = child_group.value
            assert isinstance(term_expr_prop, PropertyExpression)

            if term_type_prop.ValueDescription == WeakTermType.LINEAR.value:
                L_terms.append(f"({term_expr_prop.Value})*{domain}")
            elif term_type_prop.ValueDescription == WeakTermType.BILINEAR.value:
                a_terms.append(f"({term_expr_prop.Value})*{domain}")
            else:
                raise ValueError(
                    f"Unexpected WeakTermType: {term_type_prop.ValueDescription}"
                )

        return mdl.WeakTerm(a_terms=a_terms, L_terms=L_terms)


class WeakFormSubdomainSettings(WeakFormEquationSettings[xm.UnstructuredMeshDomain]):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(parent)

        self.description = "Subdomain Weak Terms"

    def _display_help(self) -> None:

        variable_name = self.parent_equation.variable_name

        info = f"""
Define a weak term on the specified sub-domains

The term should not include the integration domain.

{TRIAL_TEST_FUNCTION_EXPLANATION}

e.g. For a term in the PDE: ∇.(c∇({variable_name})) provide: ufl.inner(ufl.grad({variable_name}), ufl.grad({variable_name}_test))

Note:
- We have preferred ufl.inner to the multiplication operator "*" and ufl.dot.  This distinction is important if u is a complex valued field.
- This is a bi-linear term and should be marked as such
- We have not provided a term like: ufl.dx() this will be automatically added with the ids of the assigned subdomains.

{EXPRESSION_INFO}
"""
        if (
            self._parent_sim_type_prop.ValueDescription
            == mdl.SimulationType.STATIONARY.value
        ):
            info += f"""

The equation solved is:

  Σᵢ aᵢ({variable_name}, {variable_name}_test) = Σᵢ Lᵢ({variable_name}_test)
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == mdl.SimulationType.EIGENVALUE.value
        ):
            info += f"""

The equation solved is:

  Σₚ aₚ({variable_name}ᵢ, {variable_name}_test) = Σₚ Lₚ({variable_name}_test) + ∫ λᵢ{variable_name}ᵢ {variable_name}_test dω
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == mdl.SimulationType.TIME_DOMAIN.value
        ):
            info += f"""

The equation solved is:

  Σₚ aₚ({variable_name}, {variable_name}_test) = Σₚ Lₚ({variable_name}_test) + ∫ ∂{variable_name}/∂t {variable_name}_test dω
"""
        else:
            raise ValueError(
                f"Unhandled sim_type: {self._parent_sim_type_prop.ValueDescription}"
            )

        display_help(self.description, info)

    def _integral_domain_as_str(self, domain_id_map: "domain_id_map_t") -> str:
        return f"dx(({self._domain_ids_as_str(domain_id_map)}))"

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshDomain

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> mdl.WeakTerm | None:
        wk_terms = super().as_api_model(variable_name, domain_id_map)
        if wk_terms is None:
            return wk_terms

        u = self.parent_equation.variable_name
        u_shape = self.parent_equation.variable_shape
        if u_shape == mdl.VariableType.SCALAR.value:
            zero_term = "fem.Constant(_mesh_, dolfinx.default_scalar_type(0))"
        elif u_shape == mdl.VariableType.VECTOR.value:
            zero_term = "fem.Constant(_mesh_, dolfinx.default_scalar_type((0, 0, 0)))"
        else:
            raise ValueError(f"Unhandled shape for {u}: {u_shape}")

        wk_terms.L_terms.append(
            f"ufl.inner({zero_term}, {u}_test)*{self._integral_domain_as_str(domain_id_map)}"
        )

        return wk_terms


class WeakFormBoundaryFluxSettings(WeakFormEquationSettings[xm.UnstructuredMeshPatch]):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(parent)

        self.description = "Boundary Weak Terms"

    def _display_help(self) -> None:
        display_help(
            self.description,
            f"""
            Define a weak term on the specified boundaries

            The term should not include the integration domain.

            {TRIAL_TEST_FUNCTION_EXPLANATION}

            {EXPRESSION_INFO}
            """,
        )

    def _integral_domain_as_str(self, domain_id_map: "domain_id_map_t") -> str:
        return f"ds(({self._domain_ids_as_str(domain_id_map)}))"

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshPatch


class WeakFormDirichletBoundarySettings(
    EquationSettings[xm.UnstructuredMeshPatch, mdl.DirichletCondition]
):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(parent)

        self.description = "Dirichlet Boundary Settings"

        value = PropertyExpression("0")
        value.Description = "Boundary value"
        self._properties.Add("value", value)

    def _display_help(self) -> None:

        u = self.parent_equation.variable_name
        u_shape = self.parent_equation.variable_shape

        shape_info = f"{u} is a {u_shape} and so the expression should also evaluate to a {u_shape}"
        if u_shape == mdl.VariableType.VECTOR.value:
            shape_info += """
To form a vector consider using ufl.as_vector for example: ufl.as_vector((1, 2, ufl.sin(x)))
"""

        if (
            self._parent_sim_type_prop.ValueDescription
            == mdl.SimulationType.STATIONARY.value
        ):
            info = f"""
Forces the value of the dependent variable {u} to that of the provided expression on the patches assigned to this settings group.

{shape_info}

{EXPRESSION_INFO}
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == mdl.SimulationType.EIGENVALUE.value
        ):
            info = f"""
Apply a homogeneous (=0) boundary condition to the eigenfunctions {u}ᵢ on the patches assigned to this settings group.
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == mdl.SimulationType.TIME_DOMAIN.value
        ):
            info = f"""
Forces the value of the dependent variable {u} to that of the provided expression on the patches assigned to this settings group.

{shape_info}

- The temporal coordinate is defined as 'time'.
{EXPRESSION_INFO}
"""
        else:
            raise ValueError(
                f"Unhandled sim_type: {self._parent_sim_type_prop.ValueDescription}"
            )

        display_help("Dirichlet Boundary Conditions", info)

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshPatch

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> mdl.DirichletCondition | None:
        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        u_shape = self.parent_equation.variable_shape

        if (
            self._parent_sim_type_prop.ValueDescription
            == mdl.SimulationType.EIGENVALUE.value
        ):
            if u_shape == mdl.VariableType.SCALAR.value:
                value = "0"
            elif u_shape == mdl.VariableType.VECTOR.value:
                value = "ufl.as_vector([0, 0, 0])"
            else:
                raise ValueError(
                    f"Unhandle variable shape: {u_shape} for {variable_name}"
                )
        else:
            prop = self._properties.value
            assert isinstance(prop, PropertyExpression)
            value = prop.Value

        return mdl.DirichletCondition(
            domain_indices=self._domain_ids(domain_id_map),
            value_type=u_shape,
            value=value,
        )


class WeakFormEquation(Equation):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(
            parent,
            AllSubdomainSettings(self, WeakFormSubdomainSettings, "Subdomain Settings"),
            AllBoundaryFluxSettings(
                self, WeakFormBoundaryFluxSettings, "Boundary Flux Settings"
            ),
            AllDirichletSettings(
                self, WeakFormDirichletBoundarySettings, "Dirichlet Boundary Settings"
            ),
            variable_shape=mdl.VariableType.SCALAR.value,
            icon="icons/XPostProcessor/Omega.ico",
        )

        variable_type_prop = XCore.PropertyEnum(
            (mdl.VariableType.SCALAR.value, mdl.VariableType.VECTOR.value), 0
        )
        variable_type_prop.Description = "Variable Type"
        self._properties.Add("variable_type", variable_type_prop)

    def _variable_type_prop(self) -> XCore.PropertyEnum:
        prop = self._properties.variable_type
        assert isinstance(prop, XCore.PropertyEnum)
        return prop

    def _connect_signals(self) -> None:
        super()._connect_signals()

        def on_variable_type_modified(*args, **kwargs):
            self._variable_shape = self._variable_type_prop().ValueDescription

        self._variable_type_prop().OnModified.Connect(on_variable_type_modified)

    def _display_help(self) -> None:
        u = self.variable_name

        info = f"""
Define the equation governing {u} directly via its weak form.
"""

        display_help(self.description, info)
