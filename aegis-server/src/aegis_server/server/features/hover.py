from lsprotocol import types as lsp

from aegis_core.ast.features import AegisFeatureProviders, HoverParams
from aegis_core.ast.helpers import (
    BaseMetadata,
    _hash_node,
    node_location_to_range,
    retrieve_metadata,
)

from .. import AegisServer
from .helpers import (
    fetch_compilation_data,
    get_node_at_position,
)

DEBUG_AST = False


async def get_hover(ls: AegisServer, params: lsp.HoverParams):
    compiled_doc = await fetch_compilation_data(ls, params)

    if compiled_doc is None or compiled_doc.ast is None:
        return

    ast = compiled_doc.ast

    node = get_node_at_position(ast, params.position)
    text_range = node_location_to_range(node)

    if DEBUG_AST:
        return lsp.Hover(
            lsp.MarkupContent(
                lsp.MarkupKind.Markdown,
                f"Repr: `{node.__repr__()}`\n\nHash: {_hash_node(node)}\nDict: ```{retrieve_metadata(compiled_doc.resource_location, node, BaseMetadata)}```",
            ),
            text_range,
        )

    provider = compiled_doc.ctx.inject(AegisFeatureProviders).retrieve(node)

    return provider.hover(
        HoverParams(compiled_doc.ctx, node, compiled_doc.resource_location, text_range)
    )
