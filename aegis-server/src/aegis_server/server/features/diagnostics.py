import asyncio
import logging
import traceback
from threading import Timer

import lsprotocol.types as lsp
from mecha import Diagnostic

from .. import AegisServer
from ..shadows.compile_document import CompilationError
from .validate import validate_function


def tokenstream_error_to_lsp_diag(
    exec: CompilationError, source: str, filename: str | None
) -> lsp.Diagnostic:
    range = [exec.location, exec.end_location]
    # if isinstance(exec, UnexpectedToken):
    #     range = [exec.token.location, exec.token.end_location]

    trace = "\n".join(traceback.format_tb(exec.__traceback__))

    message = f"{exec.format_message() if isinstance(exec, Diagnostic) else exec.format(filename or 'unknown')}\n{type(exec).__name__}"
    logging.error(message + f"\n{trace}")
    return lsp.Diagnostic(
        range=lsp.Range(
            start=lsp.Position(
                line=max(range[0].lineno - 1, 0), character=max(range[0].colno - 1, 0)
            ),
            end=lsp.Position(
                line=max(range[1].lineno - 1, 0), character=max(range[1].colno - 1, 0)
            ),
        ),
        message=message,  # \n{exec.format(filename)}\n\n{trace}
        source=source,
    )


async def publish_diagnostics(
    ls: AegisServer,
    params: lsp.DidOpenTextDocumentParams | lsp.DidChangeTextDocumentParams,
):
    text_doc = ls.workspace.get_document(params.text_document.uri)

    with ls.context(text_doc) as ctx:
        if not ctx:
            diagnostics = []
        else:
            diagnostics = await validate_function(ctx, text_doc)
            diagnostics = [
                tokenstream_error_to_lsp_diag(d, type(ls).__name__, text_doc.filename)
                for d in diagnostics
            ]

    logging.debug(f"Sending diagnostics: {diagnostics}")

    ls.publish_diagnostics(
        params.text_document.uri,
        diagnostics,
    )


DEBOUNCE_DELAY = 0.3

_pending: dict[str, Timer] = {}

def debounce_diagnostics(
    ls: AegisServer,
    params: lsp.DidChangeTextDocumentParams
):
    uri = params.text_document.uri

    if pending := _pending.get(uri):
        pending.cancel()

    timer = Timer(DEBOUNCE_DELAY, lambda: asyncio.run(publish_diagnostics(ls, params)))
    _pending[uri] = timer
    timer.start()
