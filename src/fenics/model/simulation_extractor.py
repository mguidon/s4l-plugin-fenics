#!python3


import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import fenics.solver.driver.api_models as api_models
import XCore as xc
import XPostProcessor as xp
import XPostProPython as pp

FILENAME_SUFFIX = "000000.pvtu"
STORED_REAL_VALUED_FIELD_NAME = "f"
STORED_COMPLEX_VALUED_REAL_FIELD_NAME = "f_real"
STORED_COMPLEX_VALUED_IMAG_FIELD_NAME = "f_imag"

logger = logging.getLogger(__name__)


"""

                            Has A                         Has A                           Has A
  ┌───────────────────────┐        ┌────────────────────┐        ┌──────────────────────┐          ┌──────────────────┐
  │ PythonModuleAlgorithm ├───────►│   AlgorithmImpl    ├───────►├ SimExtractorImplBase ├─────────►│ IExtractorParent │
  │      (C++)            │        │      (Python)      │        │                      │          │                  │
  └───────────────────────┘        └────────────────────┘        └──────────────────────┘          └──────────────────┘



  ┌───────────────┐     Is A     ┌──────────────────┐
  │ AlgorithmImpl ├─────────────►│ IExtractorParent │
  └───────────────┘              └──────────────────┘



  ┌────────────────────────┐Is A ┌──────────────────────┐
  │ StationarySimExtractor ├────►│ SimExtractorImplBase │
  └────────────────────────┘     └──────────────────────┘
                                          ▲
                                          │
  ┌────────────────────────┐      Is A    │
  │ EigenvalueSimExtractor ├──────────────┘
  └────────────────────────┘

"""


class IExtractorParent(ABC):
    """
    The interface for the parent of a simulation extractor for a
    particular simulation type.  It provides the child info about the simulation
    being extracted and allows for the addition of settings and output
    ports to the parent algorithm.
    """

    @abstractmethod
    def add_property(self, name: str, property: xc.Property) -> xc.Property:
        """
        Add a property to the parent C++ class for use in the settings
        n.b. checks that this property is not already present to avoid
        an overwrite during de-serialisation.
        """
        ...

    @property
    @abstractmethod
    def input_simulation(self) -> api_models.Simulation:
        """
        Get the simulation being extracted by this extractor
        """
        ...

    @property
    @abstractmethod
    def output_files_dir(self) -> Path:
        """
        Get the directory containing the output files being extracted from
        by this extractor
        """
        ...

    @abstractmethod
    def ResizeNumberOfOutputPorts(self, num_outputs: int) -> None:
        """
        Set the number of output ports
        """
        ...


class SimExtractorImplBase(ABC):
    def __init__(self, parent: "IExtractorParent") -> None:
        super().__init__()

        self._outputs: list[xp.DataObject | None] = []
        self._parent: "IExtractorParent" = parent

    def _collect_field_names(self) -> list[str]:
        sim = self._parent.input_simulation

        eqs = [eq.variable_name for eq in sim.equations]

        pp_eqs = [pp_variable for pp_variable in sim.post_processing_expressions]

        return eqs + pp_eqs

    def _is_complex_field_type(self) -> bool:
        sim = self._parent.input_simulation
        return sim.solver_settings.field_type == api_models.FieldType.COMPLEX.value

    def _create_vtk_importer(self, filepath: Path) -> xp.VtkFieldImporter:
        extractor = xp.VtkFieldImporter()
        extractor.FileName.Value = str(filepath)
        extractor.GridUnit.Value = 4
        assert (
            extractor.GridUnit.ValueDescription == "m"
        )  # nb needs to be kept in sync with unit used in
        # Simulation._prepare_inputs()
        # i.e. exporter.Options.LengthUnit = "m"
        # TODO?: write the to the input_file?
        return extractor

    def _collect_real_field_data(
        self, extractor: xp.VtkFieldImporter, field_name: str
    ) -> xp.FloatFieldData:
        field_data = None

        for idx in range(extractor.NumberOfOutputPorts):
            data = extractor.GetOutput(idx)
            if data.Description in (STORED_REAL_VALUED_FIELD_NAME, field_name):
                field_data = data
                assert isinstance(field_data, xp.FloatFieldData)
                field_data.Quantity.Name = field_name

        assert field_data is not None

        return field_data

    def _collect_complex_field_data_from_real_vtk_data(
        self,
        real_extractor: xp.VtkFieldImporter,
        imag_extractor: xp.VtkFieldImporter,
        field_name: str,
        target_field: xp.ComplexFloatFieldData | None,
    ) -> xp.ComplexFloatFieldData:
        real_field_data = self._collect_real_field_data(real_extractor, field_name)
        imag_field_data = self._collect_real_field_data(imag_extractor, field_name)

        return self._combine_real_and_imag_field_data(
            real_field_data,
            imag_field_data,
            real_extractor.IsUpdated(0) and imag_extractor.IsUpdated(0),
            target_field,
        )

    def _combine_real_and_imag_field_data(
        self,
        real_field_data: xp.FloatFieldData,
        imag_field_data: xp.FloatFieldData,
        update_heavy_data: bool,
        target_field: xp.DataObject | None,
    ) -> xp.ComplexFloatFieldData:
        """
        _combine_real_and_imag_field_data

        update:
        target_field: Specify the target field if you want to avoid creation of a new
        FieldDataObject but rather update an existing one
        """

        if target_field is None:
            complex_field = xp.ComplexFloatFieldData()
        else:
            assert isinstance(target_field, xp.ComplexFloatFieldData)
            complex_field = target_field

        complex_field.Grid = real_field_data.Grid
        complex_field.Quantity.Name = real_field_data.Quantity.Name

        if update_heavy_data:
            complex_field.Allocate(
                real_field_data.NumberOfSnapshots,
                real_field_data.NumberOfTuples,
                real_field_data.NumberOfComponents,
            )

            complex_field.SetField(
                0, real_field_data.Field(0) + 1j * imag_field_data.Field(0)
            )

        return complex_field

    def _collect_complex_field_data_from_complex_vtk_data(
        self,
        extractor: xp.VtkFieldImporter,
        field_name: str,
        target_field: xp.DataObject | None,
    ) -> xp.ComplexFloatFieldData:
        real_field_data = None
        imag_field_data = None
        is_updated = False

        for idx in range(extractor.NumberOfOutputPorts):
            data = extractor.GetOutput(idx)
            if data.Description in (STORED_COMPLEX_VALUED_REAL_FIELD_NAME, field_name):
                real_field_data = data
                assert isinstance(real_field_data, xp.FloatFieldData)
                is_updated = extractor.IsUpdated(idx)
            elif data.Description == STORED_COMPLEX_VALUED_IMAG_FIELD_NAME:
                imag_field_data = data
                assert isinstance(imag_field_data, xp.FloatFieldData)

        assert real_field_data is not None
        assert imag_field_data is not None

        real_field_data.Quantity.Name = field_name

        return self._combine_real_and_imag_field_data(
            real_field_data, imag_field_data, is_updated, target_field
        )

    def DoCheckInputConnections(self, inputs: list[xp.AlgorithmOutput]) -> bool:
        return len(inputs) == 0

    @abstractmethod
    def DoComputeOutputAttributes(self) -> bool:
        ...

    @abstractmethod
    def DoComputeOutputData(self, index: int) -> bool:
        ...

    def GetOutputDataObject(self, output_index: int = 0) -> xp.DataObject:
        if len(self._outputs) == 0:
            self.DoComputeOutputAttributes()

        output = self._outputs[output_index]
        if output is None:
            return xp.FloatFieldData()

        return output


class StationarySimExtractorImpl(SimExtractorImplBase):
    def __init__(self, parent: "IExtractorParent") -> None:
        super().__init__(parent)
        self._extractors: list[xp.VtkFieldImporter] = []  # one per exported field

    def DoComputeOutputAttributes(self) -> bool:
        self._extractors = []

        for field_name in self._collect_field_names():
            filepath = self._parent.output_files_dir / f"{field_name}{FILENAME_SUFFIX}"
            extractor = self._create_vtk_importer(filepath)
            self._extractors.append(extractor)

        num_outputs = len(self._extractors)

        self._parent.ResizeNumberOfOutputPorts(num_outputs)

        if not all([extractor.UpdateAttributes() for extractor in self._extractors]):
            raise RuntimeError("Failed to UpdateAttributes")

        self._outputs = [None] * num_outputs

        self._update_outputs()

        return True

    def _update_outputs(self) -> bool:
        num_outputs = len(self._extractors)

        field_names = self._collect_field_names()
        assert len(field_names) == num_outputs
        assert len(self._outputs) == num_outputs

        for output_index in range(num_outputs):
            if self._is_complex_field_type():
                field_data = self._collect_complex_field_data_from_complex_vtk_data(
                    self._extractors[output_index],
                    field_names[output_index],
                    self._outputs[output_index],
                )
            else:
                field_data = self._collect_real_field_data(
                    self._extractors[output_index],
                    field_names[output_index],
                )

            self._outputs[output_index] = field_data

        return True

    def DoComputeOutputData(self, index: int) -> bool:
        if not all([extractor.Update() for extractor in self._extractors]):
            return False

        self._update_outputs()

        return True


class EigenvalueSimExtractorImpl(SimExtractorImplBase):
    def __init__(self, parent: "IExtractorParent") -> None:
        super().__init__(parent)

        self._extractors: dict[str, list[tuple[xp.VtkFieldImporter, ...]]] = {}
        # keyed by field name, is list of extractors for importing each eigenvector, each list element is either one or two VtkFieldImporters
        # If the simulation uses complex valued petsc then slepc writes all data into the eig_r and all zeros into eig_i: https://slepc.upv.es/documentation/current/docs/manualpages/EPS/EPSGetEigenvector.html

    def _get_metadata(self) -> api_models.EigenvectorsMetadata:
        with open(self._parent.output_files_dir / "eigenvectors.json") as fh:
            metadata: api_models.EigenvectorsMetadata = (
                api_models.EigenvectorsMetadata.schema().loads(fh.read())
            )

        return metadata

    def _get_eigenfunction(
        self, field_name: str, snapshot_index: int, complex_valued: bool
    ) -> xp.ComplexFloatFieldData:
        if complex_valued:
            single_eig_field_data = (
                self._collect_complex_field_data_from_complex_vtk_data(
                    self._extractors[field_name][snapshot_index][0], field_name, None
                )
            )
        else:
            single_eig_field_data = self._collect_complex_field_data_from_real_vtk_data(
                self._extractors[field_name][snapshot_index][0],
                self._extractors[field_name][snapshot_index][1],
                field_name,
                None,
            )

        return single_eig_field_data

    def DoComputeOutputAttributes(self) -> bool:
        self._extractors = {}

        metadata = self._get_metadata()
        field_names = self._collect_field_names()
        complex_valued = self._is_complex_field_type()
        output_dir = self._parent.output_files_dir

        eigenvalue_magnitudes = [
            np.abs(eig_metadata.eigenvalue_real + 1j * eig_metadata.eigenvalue_imag)
            for eig_metadata in metadata.metadata
        ]

        num_snapshots = len(metadata.metadata)
        num_outputs = len(field_names)

        self._outputs = [None] * num_outputs

        ok = True
        real_extractor = None

        for output_index, field_name in enumerate(self._collect_field_names()):
            self._extractors[field_name] = []

            for snapshot_index in range(num_snapshots):
                real_extractor = self._create_vtk_importer(
                    output_dir / f"eig{snapshot_index}_real{FILENAME_SUFFIX}"
                )
                ok = ok and real_extractor.UpdateAttributes()
                if complex_valued:
                    eig_extractor = (real_extractor,)
                else:
                    imag_extractor = self._create_vtk_importer(
                        output_dir / f"eig{snapshot_index}_imag{FILENAME_SUFFIX}"
                    )
                    ok = ok and imag_extractor.UpdateAttributes()
                    eig_extractor = (real_extractor, imag_extractor)

                self._extractors[field_name].append(eig_extractor)

            assert real_extractor is not None
            eig_field_data = xp.ComplexFloatFieldData()

            single_eig_field_data = self._get_eigenfunction(
                field_name, output_index, complex_valued
            )

            eig_field_data.Grid = single_eig_field_data.Grid
            eig_field_data.Quantity.Name = field_name
            eig_field_data.SnapshotQuantity.Name = "Eigenvalue Magnitude"
            eig_field_data.SnapshotQuantity.Unit = xp.Unit("")
            eig_field_data.Snapshots = eigenvalue_magnitudes

            self._outputs[output_index] = eig_field_data

        self._parent.ResizeNumberOfOutputPorts(num_outputs)

        return ok

    def DoComputeOutputData(self, index: int) -> bool:
        metadata = self._get_metadata()
        complex_valued = self._is_complex_field_type()

        num_snapshots = len(metadata.metadata)

        ok = True

        for output_index, field_name in enumerate(self._collect_field_names()):
            eig_field_data = None

            for snapshot_index in range(num_snapshots):
                for extractor in self._extractors[field_name][snapshot_index]:
                    ok = ok and extractor.Update()

                single_eig_field_data = self._get_eigenfunction(
                    field_name, snapshot_index, complex_valued
                )

                if snapshot_index == 0:
                    eig_field_data = self._outputs[output_index]
                    assert isinstance(eig_field_data, xp.ComplexFloatFieldData)
                    eig_field_data.Allocate(
                        num_snapshots,
                        single_eig_field_data.NumberOfTuples,
                        single_eig_field_data.NumberOfComponents,
                    )

                assert eig_field_data is not None
                eig_field_data.SetField(snapshot_index, single_eig_field_data.Field(0))

            self._outputs[output_index] = eig_field_data

        return ok


class TimeDomainSimExtractorImpl(SimExtractorImplBase):
    def __init__(self, parent: "IExtractorParent") -> None:
        super().__init__(parent)

        self._extractors: dict[str, list[xp.VtkFieldImporter]] = {}
        # keyed by field name, is list of extractors for importing each time-step,

    def _get_metadata(self) -> api_models.TimestepsMetadata:
        with open(self._parent.output_files_dir / "timesteps.json") as fh:
            metadata: api_models.TimestepsMetadata = (
                api_models.TimestepsMetadata().schema().loads(fh.read())
            )

        return metadata

    def DoComputeOutputAttributes(self) -> bool:
        self._extractors = {}

        metadata = self._get_metadata()
        field_names = self._collect_field_names()
        complex_valued = self._is_complex_field_type()

        timesteps = metadata.timesteps
        assert timesteps is not None
        num_snapshots = len(timesteps)
        num_outputs = len(field_names)

        self._outputs = [None] * num_outputs

        ok = True

        for output_index, field_name in enumerate(self._collect_field_names()):
            self._extractors[field_name] = []

            for snapshot_index in range(num_snapshots):
                filepath = (
                    self._parent.output_files_dir
                    / f"{field_name}{snapshot_index:06d}.pvtu"
                )
                extractor = self._create_vtk_importer(filepath)
                ok = ok and extractor.UpdateAttributes()
                self._extractors[field_name].append(extractor)

            if complex_valued:
                first_snapshot = self._collect_complex_field_data_from_complex_vtk_data(
                    self._extractors[field_name][0], field_name, None
                )
                output = xp.ComplexFloatFieldData()
            else:
                first_snapshot = self._collect_real_field_data(
                    self._extractors[field_name][0], field_name
                )
                output = xp.FloatFieldData()

            output.Snapshots = timesteps
            output.SnapshotQuantity.Name = "Time"
            output.SnapshotQuantity.Unit = xp.Unit("")
            output.Grid = first_snapshot.Grid
            output.Quantity.Name = field_name
            output.NumberOfComponents = first_snapshot.NumberOfComponents

            self._outputs[output_index] = output

        self._parent.ResizeNumberOfOutputPorts(num_outputs)

        return ok

    def DoComputeOutputData(self, index: int) -> bool:

        complex_valued = self._is_complex_field_type()

        ok = True
        for output_index, field_name in enumerate(self._collect_field_names()):
            for snapshot_index, extractor in enumerate(self._extractors[field_name]):
                ok = ok and extractor.Update()
                output = self._outputs[output_index]
                assert isinstance(output, (xp.FloatFieldData, xp.ComplexFloatFieldData))
                if complex_valued:
                    shapshot_field = (
                        self._collect_complex_field_data_from_complex_vtk_data(
                            self._extractors[field_name][snapshot_index],
                            field_name,
                            None,
                        )
                    )
                else:
                    shapshot_field = self._collect_real_field_data(
                        self._extractors[field_name][snapshot_index], field_name
                    )

                if (
                    output.NumberOfTuples != shapshot_field.NumberOfTuples
                    or output.NumberOfComponents != shapshot_field.NumberOfComponents
                ):
                    output.Allocate(
                        output.NumberOfSnapshots,
                        shapshot_field.NumberOfTuples,
                        shapshot_field.NumberOfComponents,
                    )

                output.SetField(snapshot_index, shapshot_field.Field(0))

        return ok


_EXTRACTOR_CLASS_MAP: dict[str, type[SimExtractorImplBase]] = {
    api_models.SimulationType.STATIONARY.value: StationarySimExtractorImpl,
    api_models.SimulationType.EIGENVALUE.value: EigenvalueSimExtractorImpl,
    api_models.SimulationType.TIME_DOMAIN.value: TimeDomainSimExtractorImpl,
}


class AlgorithmImpl(IExtractorParent):
    def __init__(self, parent: pp.PythonModuleAlgorithm) -> None:
        super().__init__()

        self._extractor: SimExtractorImplBase | None = None

        self._parent = parent
        self._parent.SetOneExecutionUpdatesAll(True)

        prop = self.add_property("results_dir", xc.PropertyString(""))
        prop.Description = "Results Dir."

        parent.Icon = "icons/XPostProcessor/field_extractor.ico"

    def add_property(self, name: str, property: xc.Property) -> xc.Property:

        if (
            existing_prop := self._parent.FindChild(name)
        ) is None:  # make sure we don't overwrite a deserialized property
            return self._parent.Add(name, property)
        else:
            assert isinstance(existing_prop, xc.Property)
            return existing_prop

    def _results_dir(self) -> Path:
        if (results_dir_prop := self._parent.FindChild("results_dir")) is None:
            raise RuntimeError("Failed to find results_dir child")

        assert isinstance(results_dir_prop, xc.PropertyString)

        assert len(results_dir_prop.Value) > 0, "Results dir was not set"

        stored_path = Path(results_dir_prop.Value)
        if stored_path.is_dir():
            results_path = stored_path
        else:  # perhaps its is a path relative to the current smash file
            results_relpath = Path(results_dir_prop.Value)

            app = xc.GetApp()
            doc = app.Document

            results_path = Path(doc.FileFolder) / results_relpath
            if not results_path.is_dir():
                raise RuntimeError(
                    f"Failed to find results_dir, tried: {stored_path} and {results_path}"
                )

        return results_path

    def _input_filepath(self) -> Path:
        input_filepath = self._results_dir() / "input_files" / "input_file.json"
        if not input_filepath.is_file():
            raise ValueError(f"Could not find: {input_filepath}")

        return input_filepath

    @property
    def input_simulation(self) -> api_models.Simulation:
        with open(self._input_filepath()) as fh:
            sim: api_models.Simulation = api_models.Simulation.schema().loads(fh.read())

        return sim

    @property
    def output_files_dir(self) -> Path:
        return self._results_dir() / "output_files"

    def ResizeNumberOfOutputPorts(self, num_outputs: int) -> None:
        self._parent.ResizeNumberOfOutputPorts(num_outputs)

    def DoCheckInputConnections(self, inputs: list[xp.AlgorithmOutput]) -> bool:
        extractor = self._get_or_create_extractor()
        return extractor.DoCheckInputConnections(inputs)

    def DoComputeOutputAttributes(self) -> bool:
        extractor = self._get_or_create_extractor()
        return extractor.DoComputeOutputAttributes()

    def DoComputeOutputData(self, index: int) -> bool:
        extractor = self._get_or_create_extractor()
        return extractor.DoComputeOutputData(index)

    def _get_or_create_extractor(self) -> "SimExtractorImplBase":
        """
        Return an extractor appropriate to the simulation type
        """

        simulation_type = self.input_simulation.simulation_type

        if simulation_type not in _EXTRACTOR_CLASS_MAP:
            raise RuntimeError(
                f"Unhandled simulation type: {self.input_simulation.simulation_type}"
            )

        extractor_cls = _EXTRACTOR_CLASS_MAP[simulation_type]

        if not isinstance(self._extractor, extractor_cls):
            self._extractor = extractor_cls(self)

        return self._extractor

    def GetOutputDataObject(self, output_index: int = 0) -> xp.DataObject:
        extractor = self._get_or_create_extractor()
        return extractor.GetOutputDataObject(output_index)
