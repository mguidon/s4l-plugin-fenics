from dataclasses import dataclass, field
from enum import Enum

from dataclasses_json import dataclass_json

try:
    # I dont want to load these into the container, but I need to import them for the type hints
    from s4l_core.simulator_plugins.base.solver.driver.api_models import (
        ApiSimulationBase,
        BaseSimulations,
    )
except ImportError:

    @dataclass_json
    @dataclass
    class ApiSimulationBase:
        pass

    @dataclass_json
    @dataclass
    class BaseSimulations:
        pass


class VariableType(Enum):
    SCALAR = "Scalar"
    VECTOR = "Vector"


@dataclass_json
@dataclass
class WeakTerm:
    # context for evals:
    # - x[0], x[1], ... available as spatial variable
    # - *TimeDomain only* 'time'
    # - all independent variables and their test: e.g. 'u' and 'u_test'
    # - global expressions from the simulation
    # - dx((n,...)) for "volume" measure over subdomains: (n, ...)
    # - ds((m, ...)) for "surface" measure over patches: (m, ...)

    a_terms: list[str]  # then a is formed by summing the result of eval-ing these
    L_terms: list[str]  # then L is formed by summing the result of eval-ing these


@dataclass_json
@dataclass
class DirichletCondition:
    # context for evals:
    # - x[0], x[1], ... available as spatial variable
    # - *TimeDomain only* 'time'
    # - ufl operators (e.g. grad, sin, exp) available
    domain_indices: list[int]
    value_type: str  # VariableType.value
    value: str


@dataclass_json
@dataclass
class Equation:
    variable_name: str
    variable_shape: str  # VariableType.value
    element_type: str
    element_degree: int
    weak_form: list[WeakTerm]
    dirichlet_conditions: list[DirichletCondition]
    initial_condition: str = ""  # ignored if empty- used by time domain problems and non-linear solvers (TODO)


class SimulationType(Enum):
    STATIONARY = "Stationary"
    EIGENVALUE = "Eigenvalue"
    TIME_DOMAIN = "Time-Domain"


class FieldType(Enum):
    REAL = "Real"
    COMPLEX = "Complex"


class SolverType(Enum):
    LINEAR = "Linear"
    NON_LINEAR = "Non-Linear (Newton)"


class LinearSolverType(Enum):
    ITERATIVE_KSP = "Iterative (KSP)"
    DIRECT_LU = "Direct (LU)"


class KSPPreConditionerType(Enum):
    EUCLID = "euclid (hypre)"
    PILUT = "parallel ilu (hypre)"
    BOOMER_AMG = "Boomer AMG (hypre)"
    NONE = "none"
    ILU = "ilu"


class KSPType(Enum):
    GMRES = "gmres"
    CG = "cg"


class RungeKuttaMethod(Enum):
    LOBATTO_IIIA = "LobattoIIIA"
    GAUSS_LEGENDRE = "Gauss Legendre"
    RADAU_IIA = "RadauIIA"


@dataclass_json
@dataclass
class SolverSettings:
    field_type: str = FieldType.REAL.value
    num_processes: int = 1
    # Stationary
    solver_type: str = SolverType.NON_LINEAR.value
    linear_solver_type: str = LinearSolverType.ITERATIVE_KSP.value
    ksp_pc: str = KSPPreConditionerType.EUCLID.value
    ksp_type: str = KSPType.GMRES.value
    ksp_rtol: float = 1e-6
    ksp_atol: float = 1e-10
    nonlin_rtol: float = 1e-6
    nonlin_atol: float = 1e-10
    # Eigenvalue
    num_eigenvals: int = 10
    target_eigenval: float = 0
    eigenval_tol: float = 1e-8
    # Time Stepping
    start_time: float = 0
    duration: float = 10
    runge_kutta_method: str = RungeKuttaMethod.LOBATTO_IIIA.value
    num_time_stepping_stages: int = 3
    initial_time_step_size: float = 1e-3
    adapt_time_step: bool = True
    max_time_step: float = 0.1
    target_error: float = 1e-3


@dataclass_json
@dataclass
class Simulation(ApiSimulationBase):
    equations: list[Equation]
    global_expressions: dict[str, str]
    post_processing_expressions: dict[str, str]
    simulation_type: str = SimulationType.STATIONARY.value
    solver_settings: SolverSettings = field(default_factory=SolverSettings)


@dataclass_json
@dataclass
class Simulations(BaseSimulations):
    simulations: list[Simulation]


@dataclass_json
@dataclass
class EigenvectorMetadata:
    eigenvalue_real: float
    eigenvalue_imag: float
    error: float


@dataclass_json
@dataclass
class EigenvectorsMetadata:
    metadata: list[EigenvectorMetadata]


@dataclass_json
@dataclass
class TimestepsMetadata:
    timesteps: list[float] | None = None
