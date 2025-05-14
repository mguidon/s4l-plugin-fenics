import logging
from typing import TYPE_CHECKING

import XCore
import XCoreModeling as xm
from s4l_core.simulator_plugins.base.model.help import display_help
from fenics.model.equation import Equation
from fenics.model.equation_settings import (
    EquationSettings,
)
from fenics.model.settings import (
    AllBoundaryFluxSettings,
    AllDirichletSettings,
    AllSubdomainSettings,
)
from fenics.model.utils import is_valid_variable_name
from fenics.solver.driver import api_models
from XCoreHeadless import DialogOptions
from XCorePython import PropertyExpression

if TYPE_CHECKING:
    from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem
    from fenics.model.fenics_simulation import (
        domain_id_map_t,
    )


logger = logging.getLogger(__name__)

GOVERNING_EQUATIONS = """
The stress tensor is related to the elastic strain εₑₗ as:

    σ = λ tr(εₑₗ)I + 2μεₑₗ

Where tr is the trace operator, I is the identity matrix, λ and μ are the Lame elasticity parameters, which are functions of Youngs Modulus E and Poisson's ratio 𝜈 for the material:

    λ = E 𝜈 / ((1 + 𝜈) (1 - 2𝜈))
    μ = E / (2 + 2𝜈)

The strain ε depends on the displacement field as:

    ε = ½ (∇w + (∇w)ᵀ)

The elastic strain is given by:

    εₑₗ = ε - αTI

Where α is the coefficient of thermal expansion for the isotropic material and T is the temperature rise with respect to the stress free temperature.
"""


class MaterialSettings(
    EquationSettings[xm.UnstructuredMeshDomain, api_models.WeakTerm]
):
    def __init__(
        self,
        parent: "TreeItem",
    ) -> None:
        super().__init__(parent)

        self.description = "Subdomain Settings"

        #

        elastic_properties = DialogOptions()
        elastic_properties.Description = "Elastic Properties"

        elastic_modulus = PropertyExpression("100e9")
        elastic_modulus.Description = "Elastic Modulus (E)"

        poisson_ratio = PropertyExpression("0.3")
        poisson_ratio.Description = "Poisson's ratio (𝜈)"

        elastic_properties.Add("elastic_modulus", elastic_modulus)
        elastic_properties.Add("poisson_ratio", poisson_ratio)

        self._properties.Add("elastic_properties", elastic_properties)

        #

        body_force = DialogOptions()
        body_force.Description = "Body Force Density"

        for axis, subscript in (("x", "ₓ"), ("y", "ᵧ"), ("z", "₂")):
            axis_expr = PropertyExpression("0")
            axis_expr.Description = f"F{subscript}"
            body_force.Add(f"F{axis}", axis_expr)

        self._properties.Add("body_force", body_force)

        #

        thermal_expansion = DialogOptions()
        thermal_expansion.Description = "Thermal Expansion"

        thermal_expansion_coefficient = PropertyExpression("1")
        thermal_expansion_coefficient.Description = "Thermal expansion coefficient"

        temperature = PropertyExpression("0")
        temperature.Description = (
            "Temperature increase with respect to the stress free temperature."
        )

        thermal_expansion.Add("temperature", temperature)
        thermal_expansion.Add(
            "thermal_expansion_coefficient", thermal_expansion_coefficient
        )

        self._properties.Add("thermal_expansion", thermal_expansion)

    def _connect_signals(self) -> None:
        super()._connect_signals()

        self._parent_sim_type_prop.OnModified.Connect(self._update_props)

        self._update_props(XCore.PropertyReal(), XCore.kPropertyModified)

    def _update_props(
        self, _: XCore.Property, mod_type: XCore.PropertyModificationTypeEnum
    ) -> None:
        if mod_type != XCore.kPropertyModified:
            return

        # no source terms for eigenvalue propblems
        for prop_name in ("body_force", "thermal_expansion"):
            prop = self._properties.FindChild(prop_name)
            assert isinstance(prop, XCore.Property)
            prop.Visible = (
                self._parent_sim_type_prop.ValueDescription
                != api_models.SimulationType.EIGENVALUE.value
            )

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshDomain

    def _display_help(self) -> None:

        w = self.parent_equation.variable_name

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = f"""
for the small displacements {w} of a linear elastic solid:

    ∇.σ({w}) = -f

Where σ is the stress tensor and f is the body force density.

{GOVERNING_EQUATIONS}

If the Non-Linear solver is selected in the Solver Settings the parameters may also be expressions of the dependent variables of this Simulation's equations.  In particular the temperature field may come from another equation.
        """
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = f"""

Solves the eigenvalue problem:

    ∇.σ({w}ᵢ) = λᵢ{w}ᵢ

Where σ is the stress tensor, {w}ᵢ is the i-th eigenfunction and λᵢ is the i-th eigenvalue.  The angular frequency ωᵢ relates to this eigenvalue as:

   λᵢ = ρωᵢ²

{GOVERNING_EQUATIONS}
        """
        else:
            raise ValueError(
                f"Unhandled sim_type: {self._parent_sim_type_prop.ValueDescription}"
            )

        display_help("Subdomain Settings", info)

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> api_models.WeakTerm | None:
        if not is_valid_variable_name(variable_name):
            raise ValueError(
                f"Tried to form api_mdl description with invalid variable name: {variable_name}"
            )

        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        u = variable_name

        E = f"{self._properties.elastic_properties.elastic_modulus.Value}"
        v = f"{self._properties.elastic_properties.poisson_ratio.Value}"

        try:  # handle the special case that these are just numbers
            E_numerical = float(E)
            v_numerical = float(v)

            lame_lambda = str(
                E_numerical * v_numerical / (1 + v_numerical) / (1 - 2 * v_numerical)
            )
            lame_mu = str(E_numerical / (2 * (1 + v_numerical)))
        except ValueError:
            lame_lambda = f"({E})*({v}) / (1+({v})) / (1- 2*({v}))"
            lame_mu = f"({E}) / (2*(1+({v})))"

        Fx_prop = self._properties.body_force.Fx
        assert isinstance(Fx_prop, PropertyExpression)
        Fx = Fx_prop.Value

        Fy_prop = self._properties.body_force.Fy
        assert isinstance(Fy_prop, PropertyExpression)
        Fy = Fy_prop.Value

        Fz_prop = self._properties.body_force.Fz
        assert isinstance(Fz_prop, PropertyExpression)
        Fz = Fz_prop.Value

        alpha_prop = self._properties.thermal_expansion.thermal_expansion_coefficient
        assert isinstance(alpha_prop, PropertyExpression)
        alpha = alpha_prop.Value

        T_prop = self._properties.thermal_expansion.temperature
        assert isinstance(T_prop, PropertyExpression)
        T = T_prop.Value

        dx = f"dx(({self._domain_ids_as_str(domain_id_map)}))"

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            # Mass term: TODO: user specified density?
            L_terms = [f"ufl.inner({u}, {u}_test) * {dx}"]
        else:
            L_terms = [
                f"ufl.dot(ufl.as_vector(({Fx}, {Fy}, {Fz})), {u}_test) * {dx}",
            ]

            if T != "0":
                L_terms.append(
                    f"ufl.inner(({lame_lambda}) * ufl.tr(({alpha})*({T})*ufl.Identity(3)) * ufl.Identity(3) + 2 * ({lame_mu}) * ({alpha})*({T})*ufl.Identity(3), ufl.sym(ufl.grad({u}_test)))* {dx}",
                )

        return api_models.WeakTerm(
            a_terms=[
                f"ufl.inner(({lame_lambda}) * ufl.nabla_div({u}) * ufl.Identity(3) + 2 * ({lame_mu}) * ufl.sym(ufl.grad({u})), ufl.sym(ufl.grad({u}_test)))* {dx}"
            ],
            L_terms=L_terms,
        )


class DisplacementBoundaryCondition(
    EquationSettings[xm.UnstructuredMeshPatch, api_models.DirichletCondition]
):
    def __init__(self, parent: "TreeItem") -> None:
        super().__init__(parent)

        self.description = "Boundary Displacement"

        displacement = DialogOptions()
        displacement.Description = "Displacement"

        for axis, subscript in (("x", "ₓ"), ("y", "ᵧ"), ("z", "₂")):
            axis_expr = PropertyExpression("0")
            axis_expr.Description = f"D{subscript}"
            displacement.Add(f"D{axis}", axis_expr)

        self._properties.Add("displacement", displacement)

    def _display_help(self) -> None:
        w = self.parent_equation.variable_name

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = f"""
Define the displacement {w} on the boundary patches assigned to this settings group
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = f"""
Apply a homogeneous (w=0) boundary condition to the eigenfunctions {w}ᵢ
on the patches assigned to this settings group.
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

        prop = self._properties.FindChild("displacement")
        assert isinstance(prop, DialogOptions)

        # Consider only homogeneous eigenvalue problem so will take 0 in this case
        prop.Visible = (
            self._parent_sim_type_prop.ValueDescription
            != api_models.SimulationType.EIGENVALUE.value
        )

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshPatch

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> api_models.DirichletCondition | None:
        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        Dx_prop = self._properties.displacement.Dx
        assert isinstance(Dx_prop, PropertyExpression)
        Dx = Dx_prop.Value

        Dy_prop = self._properties.displacement.Dy
        assert isinstance(Dy_prop, PropertyExpression)
        Dy = Dy_prop.Value

        Dz_prop = self._properties.displacement.Dz
        assert isinstance(Dz_prop, PropertyExpression)
        Dz = Dz_prop.Value

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            value = "ufl.as_vector((0, 0, 0))"
        else:
            value = f"ufl.as_vector(({Dx}, {Dy}, {Dz}))"

        return api_models.DirichletCondition(
            domain_indices=self._domain_ids(domain_id_map),
            value_type=api_models.VariableType.VECTOR.value,
            value=value,
        )


class BoundaryLoad(EquationSettings[xm.UnstructuredMeshPatch, api_models.WeakTerm]):
    def __init__(self, parent: "TreeItem") -> None:
        super().__init__(parent)

        self.description = "Boundary Load"

        traction = DialogOptions()
        traction.Description = "Traction"

        for axis, subscript in (("x", "ₓ"), ("y", "ᵧ"), ("z", "₂")):
            axis_expr = PropertyExpression("0")
            axis_expr.Description = f"T{subscript}"
            traction.Add(f"T{axis}", axis_expr)

        self._properties.Add("traction", traction)

    def _get_cell_cls(self) -> type[xm.Entity]:
        return xm.UnstructuredMeshPatch

    def _display_help(self) -> None:

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = """
Allows specification on the assigned boundary patches of:

    σ.n = T

Where σ is the stress tensor, n is the surface normal and T is commonly referred to as the traction.
"""
        elif (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = """
Enforces the homogeneous flux boundary condition, namely that the surface is stress free.
    """
        else:
            raise ValueError(
                f"Unhandled sim_type: {self._parent_sim_type_prop.ValueDescription}"
            )

        display_help("Flux Boundary Conditions", info)

    def as_api_model(
        self, variable_name: str, domain_id_map: "domain_id_map_t"
    ) -> api_models.WeakTerm | None:
        if len(self._domain_ids(domain_id_map)) == 0:
            return None

        if (
            self._parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):

            # Homogeneous boundary condition
            return api_models.WeakTerm(
                a_terms=[],
                L_terms=[],
            )

        u = variable_name

        Tx_prop = self._properties.traction.Tx
        assert isinstance(Tx_prop, PropertyExpression)
        Tx = Tx_prop.Value

        Ty_prop = self._properties.traction.Ty
        assert isinstance(Ty_prop, PropertyExpression)
        Ty = Ty_prop.Value

        Tz_prop = self._properties.traction.Tz
        assert isinstance(Tz_prop, PropertyExpression)
        Tz = Tz_prop.Value

        T = f"ufl.as_vector(({Tx}, {Ty}, {Tz}))"

        ds = f"ds(({self._domain_ids_as_str(domain_id_map)}))"

        return api_models.WeakTerm(
            a_terms=[],
            L_terms=[f"ufl.dot({T}, {u}_test) * {ds}"],
        )


class LinearElasticityEquation(Equation):
    def __init__(self, parent: "TreeItem") -> None:
        super().__init__(
            parent,
            AllSubdomainSettings(self, MaterialSettings, "Material Settings"),
            AllBoundaryFluxSettings(self, BoundaryLoad, "Boundary Load Settings"),
            AllDirichletSettings(
                self, DisplacementBoundaryCondition, "Boundary Displacement Settings"
            ),
            variable_shape=api_models.VariableType.VECTOR.value,
            variable_name="w",
            icon="icons/XModelerUI/create_solidblock.ico",
        )

    def validate(self) -> bool:
        if not super().validate():
            return False

        if (
            self.parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.TIME_DOMAIN.value
        ):
            self.status_icons = [
                "icons/TaskManager/Warning.ico",
            ]
            # TODO: Add a Runge-Kutta-Nystrom tableau
            validation_error = (
                "Time stepping a solid mechanics equation is not yet supported."
            )
            self.status_icons_tooltip += f"{validation_error}.  "
            logger.error(f"{self.description}: {validation_error}")
            return False

        return True

    def _display_help(self) -> None:
        w = self.variable_name

        if (
            self.parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.STATIONARY.value
        ):
            info = f"""
Solves for the small displacements {w} of a linear elastic solid:

    ∇.σ({w}) = -f

Where σ is the stress tensor and f is the body force density.

See the subdomain and boundary settings for further details.
"""
        elif (
            self.parent_sim_type_prop.ValueDescription
            == api_models.SimulationType.EIGENVALUE.value
        ):
            info = f"""
Solves the eigenvalue problem:

    ∇.σ({w}ᵢ) = λᵢ{w}ᵢ

Where σ is the stress tensor, {w}ᵢ is the i-th eigenfunction and λᵢ is the i-th eigenvalue.  The angular frequency ωᵢ relates to this eigenvalue as:

    λᵢ = ρωᵢ²

See the subdomain and boundary settings for further information.
"""
        else:
            raise ValueError(
                f"Unhandled sim_type: {self.parent_sim_type_prop.ValueDescription}"
            )

        display_help("Linear Elasticity Equation", info)
