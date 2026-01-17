from dataclasses import dataclass, field
from typing import Any, TypeVar

from beet import NamespaceFile
from mecha import AstNode

from ..reflection import UNKNOWN_TYPE

METADATA_KEY = "aegis_metadata"

__all__ = [
    "BaseMetadata",
    "VariableMetadata",
    "ResourceLocationMetadata",
    "attach_metadata",
    "retrieve_metadata",
]


@dataclass
class BaseMetadata:
    """
    BaseMetadata provides information to aegis_server about the AstNode.
    """


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

    type_annotation: Any = field(default=UNKNOWN_TYPE)

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


def attach_metadata(node: AstNode, metadata: BaseMetadata):
    """
    Attaches the provided metadata instance to the node

    Parameters
    ----------
    node : AstNode
        The node to attach the metadata too
    metadata : BaseMetadata
        The metadata to be attached
    """
    node.__dict__[METADATA_KEY] = metadata


T = TypeVar("T")


def retrieve_metadata(
    node: AstNode, type: tuple[type[T]] | type[T] = BaseMetadata
) -> T | None:
    """
    Retrieves the metadata attached to a node

    Parameters
    ----------
    node : AstNode
        The node to retrieve from
    type : tuple[type] | type
        The type to check the metadata for

    Returns
    -------
    BaseMetadata
        The metadata attached to the node
    None
        If not metadata is present on the node
    """
    metadata = node.__dict__.get(METADATA_KEY)

    if isinstance(metadata, type):
        return metadata

    return None
