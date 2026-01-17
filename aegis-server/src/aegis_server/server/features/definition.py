from pathlib import Path

from bolt import AstIdentifier
from lsprotocol import types as lsp
from mecha import AstResourceLocation

from aegis_core.ast.features import AegisFeatureProviders, DefinitionParams

from .. import AegisServer
from ..indexing import AegisProjectIndex, search_scope_for_binding
from .helpers import (
    fetch_compilation_data,
    get_node_at_position,
    get_representation_file,
)


async def get_definition(ls: AegisServer, params: lsp.DefinitionParams):
    compiled_doc = await fetch_compilation_data(ls, params)

    if compiled_doc is None or compiled_doc.ast is None:
        return

    node = get_node_at_position(compiled_doc.ast, params.position)

    provider = compiled_doc.ctx.inject(AegisFeatureProviders).retrieve(node)

    return provider.definition(
        DefinitionParams(compiled_doc.ctx, node, compiled_doc.resource_location)
    )

    if compiled_doc.compiled_module is None:
        return

    scope = compiled_doc.compiled_module.lexical_scope

    if isinstance(node, AstIdentifier):
        var_name = node.value

        result = search_scope_for_binding(var_name, node, scope)

        if not result:
            return

        binding, scope = result

        range = node_location_to_range(binding.origin)

        return lsp.Location(params.text_document.uri, range)
