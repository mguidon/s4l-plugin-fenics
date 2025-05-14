import XCore
import XCoreModeling as xm
from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem


class Geometry(TreeItem):
    """
    A container for mesh entities (domains or patches)
    """

    def __init__(
        self, parent: TreeItem, entity_id: str, is_expanded: bool = False
    ) -> None:
        entity = xm.GetActiveModel().LookupEntity(XCore.Uuid(entity_id))
        assert entity is not None

        super().__init__(parent, is_expanded, icon=entity.Icon)

        self._entity_id = entity_id
        self._description = (
            entity.Name
        )  #  n.b. this might change so we should watch the entity Properties

    @property
    def entity_id(self) -> str:
        return self._entity_id

    def __eq__(self, other) -> bool:
        return isinstance(other, Geometry) and self.entity_id == other.entity_id

    def __hash__(self) -> int:
        return hash(self._entity_id)

    @property
    def entity(
        self,
    ) -> xm.UnstructuredMeshDomain | xm.UnstructuredMeshPatch | None:
        entity = xm.GetActiveModel().LookupEntity(XCore.Uuid(self.entity_id))
        if not isinstance(
            entity, (xm.UnstructuredMeshDomain, xm.UnstructuredMeshPatch)
        ):
            return None
        return entity

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        raise RuntimeError("Not allowed")
