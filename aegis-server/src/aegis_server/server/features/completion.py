import builtins
import logging

from aegis_server.providers.variable import add_raw_definition, add_variable_definition
from bolt import Runtime, UnboundLocalIdentifier, UndefinedIdentifier
from lsprotocol import types as lsp
from mecha import (
    AstOption,
    BasicLiteralParser,
    Mecha,
)
from pygls.workspace import TextDocument
from tokenstream import UnexpectedEOF, UnexpectedToken

from aegis_core.ast.features import AegisFeatureProviders
from aegis_core.ast.features.provider import CompletionParams
from aegis_server.server.features.helpers import get_node_at_position

from ...server import AegisServer
from ..shadows.compile_document import CompilationError
from ..shadows.context import LanguageServerContext
from .validate import get_compilation_data

TOKEN_HINTS: dict[str, list[str]] = {
    "player_name": ["@s", "@e", "@p", "@a", "@r"],
    "coordinate": ["~ ~ ~", "^ ^ ^"],
    "item_slot": [
        "contents",
        "container.",
        "hotbar.",
        "inventory.",
        "enderchest.",
        "villager.",
        "horse.",
        "weapon",
        "weapon.mainhand",
        "weapon.offhand",
        "armor.head",
        "armor.chest",
        "armor.legs",
        "armor.feet",
        "armor.body",
        "horse.saddle",
        "horse.chest",
        "player.cursor",
        "player.crafting.\1",
    ],
}


def get_token_options(mecha: Mecha, token_type: str, value: str | None):
    # Use manually defined hints first
    if token_type in TOKEN_HINTS:
        return [lsp.CompletionItem(k) for k in TOKEN_HINTS[token_type]]

    if value == None:
        if token_type not in mecha.spec.parsers:
            return []

        parser = mecha.spec.parsers[token_type]

        # logging.debug(f"Pulling value from: {parser}")
        if isinstance(parser, BasicLiteralParser):
            node_type = parser.type
            # logging.debug(f"Node type: {node_type}")

            if issubclass(node_type, AstOption):
                # logging.debug(f"Options: {node_type.options}")
                return [
                    lsp.CompletionItem(o, kind=lsp.CompletionItemKind.Keyword)
                    for o in node_type.options
                ]
    else:
        return [lsp.CompletionItem(value)]

    return []


async def completion(ls: AegisServer, params: lsp.CompletionParams):
    text_doc = ls.workspace.get_document(params.text_document.uri)

    with ls.context(text_doc) as ctx:
        if ctx is None:
            items = None
        else:
            items = await get_completions(ctx, params.position, text_doc)

        return items


async def get_completions(
    ctx: LanguageServerContext,
    pos: lsp.Position,
    text_doc: TextDocument,
) -> lsp.CompletionList | None:
    mecha = ctx.inject(Mecha)

    if not (compiled_doc := await get_compilation_data(ctx, text_doc)):
        return None

    ast = compiled_doc.ast
    diagnostics = compiled_doc.diagnostics

    if len(diagnostics) > 0:
        return get_diag_completions(pos, mecha, ctx.inject(Runtime), diagnostics)
    elif ast is not None:
        node = get_node_at_position(ast, pos)

        provider = compiled_doc.ctx.inject(AegisFeatureProviders).retrieve(node)
        return provider.completion(
            CompletionParams(compiled_doc.ctx, node, compiled_doc.resource_location)
        )


def get_diag_completions(
    pos: lsp.Position,
    mecha: Mecha,
    runtime: Runtime,
    diagnostics: list[CompilationError],
):
    items = []
    for diagnostic in diagnostics:
        start = diagnostic.location
        end = diagnostic.end_location

        if not (start.colno <= pos.character + 1 and end.colno >= pos.character):
            continue

        if isinstance(diagnostic, UnexpectedToken) or isinstance(
            diagnostic, UnexpectedEOF
        ):
            for pattern in diagnostic.expected_patterns:
                [token_type, value] = (
                    pattern if isinstance(pattern, tuple) else [pattern, None]
                )
                items += get_token_options(mecha, token_type, value)
            
        if isinstance(diagnostic, UnboundLocalIdentifier):
            for name in mecha.spec.tree.children.keys(): #type: ignore
                items.append(lsp.CompletionItem(name, kind = lsp.CompletionItemKind.Keyword))

        if isinstance(diagnostic, UndefinedIdentifier):
            for name, variable in diagnostic.lexical_scope.variables.items():
                add_variable_definition(items, name, variable)

            for name, value in runtime.globals.items():
                add_raw_definition(items, name, value)

            for name in runtime.builtins:
                add_raw_definition(items, name, getattr(builtins, name))

    return lsp.CompletionList(False, items)
