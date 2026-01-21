from dataclasses import dataclass, field
from typing import Any, TypeVar

from .helpers import BaseMetadata, retrieve_metadata, attach_metadata, clear_metadata
from beet import NamespaceFile
from mecha import AstNode

from ..reflection import UNKNOWN_TYPE
from ..reflection.type_representation import TypeRepresentation


METADATA_KEY = "aegis_metadata"

__all__ = [
    "VariableMetadata",
    "ResourceLocationMetadata",
    "retrieve_metadata",
    "attach_metadata",
    "clear_metadata"
]



@dataclass
class VariableMetadata(BaseMetadata):
    """
    VariableMetadata provides information to aegis_server about a node representing a Bolt variable

    Attributes
    ----------
    type_annotation : Any
        The python type that the node represents

    documentation : str
        The documentation string for the node
    """

    type_annotation: TypeRepresentation = field(default_factory=lambda: UNKNOWN_TYPE)

    documentation: str | None = field(default=None)


@dataclass
class ResourceLocationMetadata(BaseMetadata):
    """
    ResourceLocationMetadata provides information to aegis_server about a node representing a resource location node

    Attributes
    ----------
    respresents : str | type[NamespaceFile] | None
        The registry or type of File the resource location represents

    unresolved_path : str | None
        The unresolved path of the string, ex. ~/foo
    """

    represents: str | type[NamespaceFile] | None = field(default=None)

    unresolved_path: str | None = field(default=None)



