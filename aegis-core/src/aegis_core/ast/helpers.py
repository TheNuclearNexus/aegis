from typing import Iterable

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
