from bolt import AstIdentifier, AstTargetIdentifier
from lsprotocol import types as lsp
from mecha import AstNode

from aegis_core.ast.helpers import node_location_to_range, offset_location

from .. import AegisServer
from ..indexing import search_scope_for_binding
from .helpers import (
    fetch_compilation_data,
    get_node_at_position,
)


async def rename_variable(ls: AegisServer, params: lsp.RenameParams):
    compiled_doc = await fetch_compilation_data(ls, params)

    if compiled_doc is None or compiled_doc.compiled_module is None:
        return

    ast = compiled_doc.compiled_module.ast
    scope = compiled_doc.compiled_module.lexical_scope

    node = get_node_at_position(ast, params.position)
    if isinstance(node, AstIdentifier) or isinstance(node, AstTargetIdentifier):
        var_name = node.value

        if not (result := search_scope_for_binding(var_name, node, scope)):
            return
        binding, _ = result

        edits = []

        origin_node = AstNode(
            binding.origin.location,
            offset_location(binding.origin.location, len(var_name)),
        )

        edits.append(lsp.TextEdit(node_location_to_range(origin_node), params.new_name))

        for reference in binding.references:
            edits.append(
                lsp.TextEdit(node_location_to_range(reference), params.new_name)
            )

        return lsp.WorkspaceEdit(changes={params.text_document.uri: edits})
