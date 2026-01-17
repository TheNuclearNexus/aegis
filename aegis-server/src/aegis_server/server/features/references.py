from pathlib import Path

from bolt import AstIdentifier, AstTargetIdentifier
from lsprotocol import types as lsp
from mecha import AstResourceLocation

from aegis_core.ast.features import AegisFeatureProviders, ReferencesParams

from .. import AegisServer
from ..indexing import AegisProjectIndex, search_scope_for_binding
from .helpers import (
    fetch_compilation_data,
    get_node_at_position,
    get_representation_file,
)


async def get_references(ls: AegisServer, params: lsp.ReferenceParams):
    compiled_doc = await fetch_compilation_data(ls, params)

    if compiled_doc is None or compiled_doc.ast is None:
        return

    node = get_node_at_position(compiled_doc.ast, params.position)

    provider = compiled_doc.ctx.inject(AegisFeatureProviders).retrieve(node)

    return provider.references(
        ReferencesParams(compiled_doc.ctx, node, compiled_doc.resource_location)
    )

    if isinstance(node, AstResourceLocation):
        if not (represents := get_representation_file(project_index, node)):
            return

        path = node.get_canonical_value()

    if compiled_doc.compiled_module is None:
        return

    scope = compiled_doc.compiled_module.lexical_scope

    if isinstance(node, AstIdentifier) or isinstance(node, AstTargetIdentifier):
        var_name = node.value

        if not (result := search_scope_for_binding(var_name, node, scope)):
            return

        binding, _ = result

        locations = []
        for reference in binding.references:
            range = node_location_to_range(reference)
            locations.append(lsp.Location(params.text_document.uri, range))

        return locations
