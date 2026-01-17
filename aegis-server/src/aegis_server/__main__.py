import argparse
import asyncio
import logging

from aegis_core.indexing.project_index import AegisProjectIndex
from lsprotocol import types as lsp

from . import __version__

from .server import AegisServer
from .server.features import hover as hover_feature
from .server.features.completion import completion
from .server.features.definition import get_definition
from .server.features.diagnostics import publish_diagnostics
from .server.features.hover import get_hover
from .server.features.references import get_references
from .server.features.rename import rename_variable
from .server.features.semantics import (
    TOKEN_MODIFIERS,
    TOKEN_TYPES,
    semantic_tokens,
)


def create_server():
    server = AegisServer("aegis-server", __version__)

    @server.thread()
    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(ls: AegisServer, params: lsp.DidChangeTextDocumentParams):
        asyncio.run(publish_diagnostics(ls, params))

    @server.thread()
    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def did_open(ls: AegisServer, params: lsp.DidOpenTextDocumentParams):
        asyncio.run(publish_diagnostics(ls, params))

    @server.thread()
    @server.feature(
        lsp.TEXT_DOCUMENT_COMPLETION,
        lsp.CompletionOptions(trigger_characters=[" ", "/", "."]),
    )
    def get_completion(ls: AegisServer, params: lsp.CompletionParams):
        return asyncio.run(completion(ls, params))

    @server.feature(lsp.INITIALIZED)
    def initialized(ls: AegisServer, params: lsp.InitializedParams):
        ls.setup_workspaces()

    @server.feature(lsp.WORKSPACE_DID_CHANGE_WORKSPACE_FOLDERS)
    def folders_changed(ls: AegisServer, params: lsp.DidChangeWorkspaceFoldersParams):
        ls.setup_workspaces()

    @server.thread()
    @server.feature(
        lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
        lsp.SemanticTokensLegend(
            token_types=list(TOKEN_TYPES.keys()),
            token_modifiers=list(TOKEN_MODIFIERS.keys()),
        ),
    )
    def semantic_tokens_full(ls: AegisServer, params: lsp.SemanticTokensParams):
        return asyncio.run(semantic_tokens(ls, params))

    @server.thread()
    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def definition(ls: AegisServer, params: lsp.DefinitionParams):
        return asyncio.run(get_definition(ls, params))

    @server.thread()
    @server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
    def references(ls: AegisServer, params: lsp.ReferenceParams):
        return asyncio.run(get_references(ls, params))

    @server.thread()
    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def hover(ls: AegisServer, params: lsp.HoverParams):
        return asyncio.run(get_hover(ls, params))

    @server.thread()
    @server.feature(lsp.TEXT_DOCUMENT_RENAME)
    def rename(ls: AegisServer, params: lsp.RenameParams):
        return asyncio.run(rename_variable(ls, params))

    @server.command("mecha.server.dumpIndices")
    def dump(ls: AegisServer, *args):
        for _, i in ls._instances.values():
            index = i.inject(AegisProjectIndex)
            ls.show_message_log(index.dump())

    @server.command("mecha.server.toggleASTDebug")
    def toggle_ast_debug(ls: AegisServer, *args):
        hover_feature.DEBUG_AST = not hover_feature.DEBUG_AST

    return server


def add_arguments(parser: argparse.ArgumentParser):
    parser.description = "simple json server example"

    parser.add_argument("--tcp", action="store_true", help="Use TCP server")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind to this address")
    parser.add_argument("--port", type=int, default=2087, help="Bind to this port")
    parser.add_argument(
        "--site",
        type=str,
        default=[],
        nargs="*",
        help="Sites to look for python packages",
    )
    parser.add_argument(
        "--debug_ast",
        type=bool,
        default=False,
        help="Show the AST node for the hovered token",
    )


def main():
    print("Starting Aegis Server")
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    aegis_server = create_server()
    hover_feature.DEBUG_AST = args.debug_ast

    aegis_server.set_sites(args.site if args.site is not None else [])

    if args.tcp:
        aegis_server.start_tcp(args.host, args.port)
    elif args.ws:
        aegis_server.start_ws(args.host, args.port)
    else:
        aegis_server.start_io()

    aegis_server._kill()


if __name__ == "__main__":
    main()
