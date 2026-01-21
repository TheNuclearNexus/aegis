from dataclasses import dataclass
from typing import Iterable, TypeVar

import lsprotocol.types as lsp
from mecha import AstNode
from tokenstream import SourceLocation


def node_location_to_range(node: AstNode | Iterable[SourceLocation]):
    if isinstance(node, AstNode):
        location = node.location
        end_location = node.end_location
    else:
        location, end_location = node

    return lsp.Range(
        start=location_to_position(location), end=location_to_position(end_location)
    )


def node_start_to_range(node: AstNode):
    start = location_to_position(node.location)
    end = lsp.Position(line=start.line, character=start.character + 1)

    return lsp.Range(start=start, end=end)


def location_to_position(location: SourceLocation) -> lsp.Position:
    return lsp.Position(
        line=max(location.lineno - 1, 0),
        character=max(location.colno - 1, 0),
    )


def offset_location(location: SourceLocation, offset):
    return SourceLocation(
        location.pos + offset, location.lineno, location.colno + offset
    )

@dataclass
class BaseMetadata:
    """
    BaseMetadata provides information to aegis_server about the AstNode.
    """

_metadata_storage: dict[str, dict[int, BaseMetadata]] = dict()

def clear_metadata(resource_location: str):
    _metadata_storage[resource_location] = dict()

def _hash_node(node: AstNode):
    return hash(hash(type(node).__name__) + hash(node.location.pos))

def attach_metadata(resource_location: str, node: AstNode|int, metadata: BaseMetadata):
    """
    Attaches the provided metadata instance to the node

    Parameters
    ----------
    node : AstNode|int
        The node to attach the metadata too
    metadata : BaseMetadata
        The metadata to be attached
    """

    storage = _metadata_storage.setdefault(resource_location, {})
    
    node_hash = _hash_node(node) if not isinstance(node, int) else node

    storage[node_hash] = metadata

T = TypeVar("T")

def retrieve_metadata(
    resource_location: str, node: AstNode|int, type: tuple[type[T]] | type[T] = BaseMetadata
) -> T | None:
    """
    Retrieves the metadata attached to a node

    Parameters
    ----------
    node : AstNode | int
        The node or hash to retrieve from
    type : tuple[type] | type
        The type to check the metadata for

    Returns
    -------
    BaseMetadata
        The metadata attached to the node
    None
        If not metadata is present on the node
    """

    storage = _metadata_storage.setdefault(resource_location, {})
    
    node_hash = _hash_node(node) if not isinstance(node, int) else node

    metadata = storage.get(node_hash)

    if isinstance(metadata, type):
        return metadata

    return None

