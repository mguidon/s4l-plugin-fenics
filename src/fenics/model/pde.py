import logging
from typing import TYPE_CHECKING

import XCore
import XCoreModeling as xm
from s4l_core.simulator_plugins.base.model.help import display_help
from fenics.model.equation import Equation
from fenics.model.equation_settings import (
    EquationSettings,
)
from fenics.model.expressions import EXPRESSION_INFO
from fenics.model.settings import (
    AllBoundaryFluxSettings,
    AllDirichletSettings,
    AllSubdomainSettings,
)
from fenics.model.utils import is_valid_variable_name
from fenics.solver.driver import api_models
from XCore import Property
from XCoreHeadless import DialogOptions
from XCorePython import PropertyExpression

if TYPE_CHECKING:
    from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem
    from fenics.model.fenics_simulation import (
        domain_id_map_t,
    )


logger = logging.getLogger(__name__)


class PdeSubdomainSettings(
    EquationSettings[xm.UnstructuredMeshDomain, api_models.WeakTerm]
):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(parent)

        self.description = "Subdomain Settings"

        #

        diffusion_coeff = PropertyExpression("1")
        diffusion_coeff.Description = "Diffusion Coefficient: c"

        source = PropertyExpression("0")
        source.Description = "Source Term: f"

        linear = PropertyExpression("0")
        linear.Description = "Linear Term: l"

        self._properties.Add("diffusion_coeff", diffusion_coeff)
        self._properties.Add("source", source)
        self._properties.Add("linear", linear)

        #

        divergence = DialogOptions()
        divergence.Description = "divergence (M)"

        for axis, subscript in (("x", "ₓ"), ("y", "ᵧ"), ("z", "₂")):
            axis_expr = PropertyExpression("0")
            axis_expr.Description = f"M{subscript}"
            divergence.Add(f"M{axis}", axis_expr)

        self._properties.Add("divergence", divergence)

    def _connect_signals(self) -> None:
        super()._connect_signals()

        self._parent_sim_type_prop.OnModified.Connect(self._update_props)

        self._update_props(XCore.PropertyReal(), XCore.kPropertyModified)

    def _display_help(self) -> None:
        SUBDOMAIN_PARAMETER_INFO = """
The parameters, c, l, M and f may be functions of position or the Global Expressions.
"""

        u = self.parent_equation.variable_name

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = f"""
Solves for the dependent variable {u}(x):

    ∇.(c∇{u}) + l{u} + ∇.M = -f

On the assigned subdomains.

{SUBDOMAIN_PARAMETER_INFO}

If the Non-Linear solver is selected in the Solver Settings the parameters may also be expressions of the dependent variables of this Simulation's equations (e.g. Poisson-Boltzmann term: f = ufl.sinh(u))

{EXPRESSION_INFO}
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = f"""

Solves the eigenvalue problem:

    ∇.(c∇{u}ᵢ) + l{u}ᵢ = λᵢ{u}ᵢ

{SUBDOMAIN_PARAMETER_INFO}

{EXPRESSION_INFO}
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.TIME_DOMAIN.value
        ):
            info = f"""

Solves the time domain problem problem:

    ∇.(c∇{u}) + l{u} + ∇.M =  -f + ∂{u}/∂t

{SUBDOMAIN_PARAMETER_INFO}

- The temporal coordinate is defined as 'time'.
{EXPRESSION_INFO}
"""

        else:
            raise ValueError(
                f"Unhandled sim_type: {self._parent_sim_type_prop.ValueDescription}"
            )

        display_help("Subdomain Settings", info)

    def _update_props(
        self, _: XCore.Property, mod_type: XCore.PropertyModificationTypeEnum
    ) -> None:
        if mod_type != XCore.kPropertyModified:
            return

        # no source terms for eigenvalue problems
        for prop_name in ("source", "divergence"):
            prop = self._properties.FindChild(prop_name)
            assert isinstance(prop, Property)
            prop.Visible = (
                self._parent_sim_type_prop.ValueDescription
                != api_models.SimulationType.EIGENVALUE.value
            )

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshDomain

    # accessors: testing

    @property
    def diffusion_coeff(self) -> str:
        prop = self._properties.diffusion_coeff
        assert isinstance(prop, PropertyExpression)
        return prop.Value

    @diffusion_coeff.setter
    def diffusion_coeff(self, value: str) -> None:
        self._properties.diffusion_coeff.Value = value

    @property
    def source(self) -> str:
        prop = self._properties.source
        assert isinstance(prop, PropertyExpression)
        return prop.Value

    @source.setter
    def source(self, value: str) -> None:
        prop = self._properties.source
        assert isinstance(prop, PropertyExpression)
        prop.Value = value

    @property
    def linear(self):
        prop = self._properties.linear
        assert isinstance(prop, PropertyExpression)
        return prop.Value

    @linear.setter
    def linear(self, value) -> None:
        prop = self._properties.linear
        assert isinstance(prop, PropertyExpression)
        prop.Value = value

    @property
    def My(self) -> str:
        My_prop = self._properties.divergence.My
        assert isinstance(My_prop, PropertyExpression)
        return My_prop.Value

    @My.setter
    def My(self, value: str) -> None:
        My_prop = self._properties.divergence.My
        assert isinstance(My_prop, PropertyExpression)
        My_prop.Value = value

    #

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> api_models.WeakTerm | None:
        if not is_valid_variable_name(variable_name):
            raise ValueError(
                f"Tried to form api_mdl description with invalid variable name: {variable_name}"
            )

        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        c = self.diffusion_coeff
        f = self.source
        lin = self.linear
        V = variable_name

        #

        Mx_prop = self._properties.divergence.Mx
        assert isinstance(Mx_prop, PropertyExpression)
        Mx = Mx_prop.Value

        My_prop = self._properties.divergence.My
        assert isinstance(My_prop, PropertyExpression)
        My = My_prop.Value

        Mz_prop = self._properties.divergence.Mz
        assert isinstance(Mz_prop, PropertyExpression)
        Mz = Mz_prop.Value

        dx = f"dx(({self._domain_ids_as_str(domain_id_map)}))"

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            L_terms = [
                f"ufl.inner({V}, {V}_test) * {dx}"
            ]  # Mass term: TODO: allow user specified mass property
        else:
            L_terms = [
                f"ufl.inner(fem.Constant(_mesh_, dolfinx.default_scalar_type(0)), {V}_test)*{dx}",  # this term is here to ensure we get a rhs (i.e. not 0*dx which is then dropped)
                f"ufl.inner(-1*({f}), {V}_test)*{dx}",
                f"ufl.inner(ufl.as_vector([{Mx}, {My}, {Mz}]), ufl.grad({V}_test)) * {dx}",
            ]

        return api_models.WeakTerm(
            a_terms=[
                f"ufl.inner(-1*({c}) * ufl.grad({V}), ufl.grad({V}_test)) * {dx}",
                f"ufl.inner(({lin})*{V}, {V}_test) * {dx}",
            ],
            L_terms=L_terms,
        )


class PdeDirichletBoundarySettings(
    EquationSettings[xm.UnstructuredMeshPatch, api_models.DirichletCondition]
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

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = f"""
 Forces the value of the dependent variable {u} to that of the provided expression
 on the patches assigned to this settings group.

{EXPRESSION_INFO}
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = f"""
Apply a homogeneous (=0) boundary condition to the eigenfunctions {u}ᵢ
on the patches assigned to this settings group.
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.TIME_DOMAIN.value
        ):
            info = f"""
Forces the value of the dependent variable {u} to that of the provided expression
on the patches assigned to this settings group.

- The temporal coordinate is defined as 'time'.
{EXPRESSION_INFO}
"""
        else:
            raise ValueError(
                f"Unhandled sim_type: {self._parent_sim_type_prop.ValueDescription}"
            )

        display_help("Dirichlet Boundary Conditions", info)

    def _connect_signals(self) -> None:
        super()._connect_signals()

        self._parent_sim_type_prop.OnModified.Connect(self._update_props)

        self._update_props(XCore.PropertyReal(), XCore.kPropertyModified)

    def _update_props(
        self, _: XCore.Property, mod_type: XCore.PropertyModificationTypeEnum
    ) -> None:
        if mod_type != XCore.kPropertyModified:
            return

        prop = self._properties.FindChild("value")
        assert isinstance(prop, PropertyExpression)

        # Consider only homogeneous eigenvalue problem so will take 0 in this case
        prop.Visible = (
            self._parent_sim_type_prop.ValueDescription
            != api_models.SimulationType.EIGENVALUE.value
        )

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshPatch

    # accessors: testing

    @property
    def value(self) -> str:
        prop = self._properties.value
        assert isinstance(prop, PropertyExpression)
        return prop.Value

    @value.setter
    def value(self, value: str) -> None:
        prop = self._properties.value
        assert isinstance(prop, PropertyExpression)
        prop.Value = value

    #

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> api_models.DirichletCondition | None:
        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            value = "0"
        else:
            value = self.value

        return api_models.DirichletCondition(
            domain_indices=self._domain_ids(domain_id_map),
            value_type=api_models.VariableType.SCALAR.value,
            value=value,
        )


class PdeBoundaryFluxSettings(
    EquationSettings[xm.UnstructuredMeshPatch, api_models.WeakTerm]
):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(parent)

        self.description = "Flux Boundary Conditions"

        flux0 = PropertyExpression("0")
        flux0.Description = "Boundary flux: q₀"
        self._properties.Add("flux0", flux0)

        flux1 = PropertyExpression("0")
        flux1.Description = "Proportional flux: q₁"
        flux1.ToolTip = "Boundary flux proportional to dependent variable"
        self._properties.Add("flux1", flux1)

    def _connect_signals(self) -> None:
        super()._connect_signals()

        self._parent_sim_type_prop.OnModified.Connect(self._update_props)

        self._update_props(XCore.PropertyReal(), XCore.kPropertyModified)

    def _display_help(self) -> None:
        u = self.parent_equation.variable_name

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = f"""
For the dependent variable {u} enforces on the assigned boundary patches:

    c∇u.n = q₀ + uq₁

Where c is specified from the subdomain on which the patch lies.  q₀ and q₁ may be functions of position and the global expressions.

If the Non Linear solver is selected in the Solver Settings they may also be functions of the dependent variables.

{EXPRESSION_INFO}
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = f"""
For the dependent variable {u} enforces the homogeneous boundary condition on the assigned boundary patches:

    c∇u.n = uq₁

Where c is specified from the subdomain on which the patch lies.  q₁ may be a function of position and the global expressions.

{EXPRESSION_INFO}
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.TIME_DOMAIN.value
        ):
            info = f"""
For the dependent variable {u} enforces the homogeneous boundary condition on the assigned boundary patches:

    c∇u.n = q₀ + uq₁

Where c is specified from the subdomain on which the patch lies.  q₁ may be a function of position, time and the global expressions.

- The temporal coordinate is defined as 'time'.
{EXPRESSION_INFO}
"""
        else:
            raise ValueError(
                f"Unhandled sim_type: {self._parent_sim_type_prop.ValueDescription}"
            )

        display_help("Flux Boundary Conditions", info)

    def _update_props(
        self, _: XCore.Property, mod_type: XCore.PropertyModificationTypeEnum
    ) -> None:
        if mod_type != XCore.kPropertyModified:
            return

        flux0_prop = self._properties.FindChild("flux0")
        assert isinstance(flux0_prop, PropertyExpression)

        # Consider only homogeneous eigenvalue problem so will take 0 in this case
        flux0_prop.Visible = (
            self._parent_sim_type_prop.ValueDescription
            != api_models.SimulationType.EIGENVALUE.value
        )

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshPatch

    # accessors: testing

    @property
    def flux0(self) -> str:
        prop = self._properties.flux0
        assert isinstance(prop, PropertyExpression)
        return prop.Value

    @flux0.setter
    def flux0(self, flux0: str) -> None:
        prop = self._properties.flux0
        assert isinstance(prop, PropertyExpression)
        prop.Value = flux0

    @property
    def flux1(self) -> str:
        prop = self._properties.flux1
        assert isinstance(prop, PropertyExpression)
        return prop.Value

    @flux1.setter
    def flux1(self, flux1: str) -> None:
        prop = self._properties.flux1
        assert isinstance(prop, PropertyExpression)
        prop.Value = flux1

    #

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> api_models.WeakTerm | None:
        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        f0 = self.flux0
        f1 = self.flux1
        V = variable_name

        ds = f"ds(({self._domain_ids_as_str(domain_id_map)}))"

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            L_terms = []
        else:
            L_terms = [f"ufl.inner(-1*({f0}), {V}_test)*{ds}"]

        return api_models.WeakTerm(
            a_terms=[f"ufl.inner(({f1})*{V}, {V}_test)*{ds}"],
            L_terms=L_terms,
        )


class PDE(Equation):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(
            parent,
            AllSubdomainSettings(self, PdeSubdomainSettings, "Subdomain Settings"),
            AllBoundaryFluxSettings(
                self, PdeBoundaryFluxSettings, "Boundary Flux Settings"
            ),
            AllDirichletSettings(
                self, PdeDirichletBoundarySettings, "Dirichlet Boundary Settings"
            ),
            variable_shape=api_models.VariableType.SCALAR.value,
            icon="icons/XPostProcessor/PhysicalScalarQuantity.ico",
        )

    def _display_help(self) -> None:
        u = self.variable_name

        if (
            self.parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = f"""
Solve the classical scalar PDE:

    ∇.(c∇{u}) + l{u}  = -f - ∇.M

See the subdomain and boundary settings for further information.
"""
        elif (
            self.parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = f"""
Solves the eigenvalue problem:

    ∇.(c∇{u}ᵢ) + l{u}ᵢ = λᵢ{u}ᵢ

See the subdomain and boundary settings for further information.
"""
        elif (
            self.parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.TIME_DOMAIN.value
        ):
            info = f"""

Solves the time domain problem problem:

    ∇.(c∇{u}) + l{u}  =  -f - ∇.M + ∂{u}/∂t

See the subdomain and boundary settings for further information.
"""
        else:
            raise ValueError(
                f"Unhandled sim_type: {self.parent_sim_type_prop.ValueDescription}"
            )

        display_help("PDE Equation", info)
