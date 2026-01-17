from typing import Any, cast

from beet import NamespaceFile
from lsprotocol import types as lsp
from mecha import AstNode, AstResourceLocation
from tokenstream import SourceLocation

from aegis_core.ast.metadata import ResourceLocationMetadata, retrieve_metadata

from .. import AegisServer
from .validate import get_compilation_data


def get_representation_file(node: AstResourceLocation):
    metadata = retrieve_metadata(node, ResourceLocationMetadata)

    if not metadata:
        return None

    if not (represents := cast(type[NamespaceFile] | str | None, metadata.represents)):
        return None

    if isinstance(represents, str):
        return None

    return represents


async def fetch_compilation_data(ls: AegisServer, params: Any):
    text_doc = ls.workspace.get_document(params.text_document.uri)
    with ls.context(text_doc) as ctx:

        if ctx is None:
            return None

        compiled_doc = await get_compilation_data(ctx, text_doc)
        return compiled_doc


def get_node_at_position(root: AstNode, pos: lsp.Position):
    target = SourceLocation(0, pos.line + 1, pos.character + 1)
    nearest_node = root
    for node in root.walk():
        start = node.location
        end = node.end_location

        if not (start.colno <= target.colno and end.colno >= target.colno):
            continue

        if not (start.lineno == target.lineno and end.lineno == target.lineno):
            continue

        if (
            start.pos >= nearest_node.location.pos
            and end.pos <= nearest_node.end_location.pos
        ):
            nearest_node = node

    return nearest_node

