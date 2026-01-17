import asyncio
import logging
import multiprocessing
import os
import signal
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePath
from typing import Any, TypeVar

from beet import Context, DataPack, Function, NamespaceFile, PackLoadUrl, TextFileBase
from beet.core.utils import extra_field, required_field
from bolt import Module, Runtime
from mecha import (
    AbstractNode,
    AstChildren,
    AstNode,
    AstRoot,
)
from mecha import CompilationError as McCompilationError
from mecha import (
    CompilationUnit,
    Diagnostic,
    DiagnosticCollection,
    Dispatcher,
    Mecha,
    MutatingReducer,
    rule,
    dispatch
)
from mecha.ast import AstError
from mecha.contrib.nested_location import (
    NestedLocationResolver,
    NestedLocationTransformer,
)
from pygls.workspace import TextDocument
from tokenstream import InvalidSyntax, SourceLocation, TokenStream

from ..indexing import AegisProjectIndex, Indexer
from ..shadows.compile_document import (
    COMPILATION_RESULTS,
    CompilationError,
    CompiledDocument,
)
from ..shadows.context import LanguageServerContext

SUPPORTED_EXTENSIONS = [Function.extension, Module.extension]
T = TypeVar("T", bound=AstNode)


async def get_compilation_data(ctx: LanguageServerContext, text_doc: TextDocument):
    await COMPILATION_LOCK.acquire()
    COMPILATION_LOCK.release()

    resource = ctx.path_to_resource.get(
        os.path.normcase(os.path.normpath(text_doc.path))
    )

    if resource and resource[0] in COMPILATION_RESULTS:
        return COMPILATION_RESULTS[resource[0]]

    await validate_function(ctx, text_doc)

    resource = resource or ctx.path_to_resource.get(text_doc.path)

    if resource is None:
        return None

    # logging.debug(COMPILATION_RESULTS)
    return COMPILATION_RESULTS[resource[0]]


COMPILATION_LOCK = asyncio.Semaphore()


@asynccontextmanager
async def semaphore(lock: asyncio.Semaphore):
    await lock.acquire()
    try:
        yield
        lock.release()
    except Exception as ex:
        lock.release()
        raise ex
    finally:
        lock.release()


async def validate_function(
    ctx: LanguageServerContext, text_doc: TextDocument
) -> list[CompilationError]:

    path = os.path.normcase(os.path.normpath(text_doc.path))
    logging.debug(f"Queuing compilation of `{path}`")
    async with semaphore(COMPILATION_LOCK):

        logging.debug(f"Starting compilation of `{path}`")

        if path not in ctx.path_to_resource:
            if not try_to_mount_file(ctx, path):
                logging.debug("File not in workspaces")
                return []

        location, file = ctx.path_to_resource[path]

        if not isinstance(file, Function) and not isinstance(file, Module):
            COMPILATION_RESULTS[location] = CompiledDocument(
                ctx, location, None, [], None, None
            )
            logging.debug("File is not a function or module.")
            return []

        try:
            async with asyncio.timeout(10):
                compiled_doc = await parse_function(
                    ctx,
                    location,
                    text_doc.path,
                    type(file)(text_doc.source, text_doc.path),
                )

            COMPILATION_RESULTS[location] = compiled_doc
            res = compiled_doc.diagnostics

        except TimeoutError as ex:
            logging.debug(f"Compilation took longer than 10 seconds, aborting\n{ex}")
            res = []

    return res


def try_to_mount_file(ctx: LanguageServerContext, file_path: str):
    """Try to mount a given file path to the context. True if the file was successfully mounted"""

    _, file_extension = os.path.splitext(file_path)
    if not file_extension in SUPPORTED_EXTENSIONS:
        return False

    load_options = ctx.project_config.data_pack.load
    prefix = None
    for entry in load_options.entries():
        # File can't be relative to a url
        if isinstance(entry, PackLoadUrl):
            continue

        if isinstance(entry, dict):
            for key, paths in entry.items():
                for path in paths.entries():
                    # File can't be relative to a url
                    if isinstance(path, PackLoadUrl):
                        continue

                    if PurePath(file_path).is_relative_to(path):
                        relative = PurePath(file_path).relative_to(path)
                        prefix = str(key / relative)
                        break
        elif PurePath(file_path).is_relative_to(entry):
            relative = PurePath(file_path).relative_to(entry)
            prefix = str(relative)

    if prefix == None:
        return False

    try:
        temp = DataPack()
        temp.mount(prefix, file_path)
        for [location, file] in temp.all():
            if not (isinstance(file, Function) or isinstance(file, Module)):
                continue

            path = os.path.normpath(file.ensure_source_path())
            path = os.path.normcase(path)
            ctx.path_to_resource[path] = (location, file)
            ctx.data[type(file)][location] = file

        logging.debug(f"Mounted {file_path} to {location}")
        return True
    except Exception as exc:
        logging.error(f"Failed to mount {file_path}, reloading datapack,\n{exc}")

    return False


@dataclass
class ErrorAccumulator(MutatingReducer):
    _errors: list[InvalidSyntax] = extra_field(default_factory=list)
    resource_location: str = required_field()
    filename: str | None = required_field()
    file_instance: TextFileBase[Any] = required_field()

    @rule(AstError)
    def error(self, error: AstError):
        logging.error(error.error)
        self._errors.append(error.error)

        return None

    def collect(self, root: T | None) -> tuple[T | None, list[InvalidSyntax]]:
        if root is None:
            return (root, [])

        root = self.__call__(root)

        return (root, self._errors)


Node = TypeVar("Node", bound=AbstractNode)


async def parse_function(
    ctx: LanguageServerContext,
    resource_location: str,
    source_path: str,
    file_instance: Function | Module,
) -> CompiledDocument:

    start = time.time()
    ast, errors = await compile(ctx, resource_location, source_path, file_instance)
    logging.debug(f"Compilation for {source_path} took {time.time() - start}s")

    # # Parse the stream
    mecha = ctx.inject(Mecha)
    runtime = ctx.inject(Runtime)

    compilation_unit = mecha.database[file_instance]
    compiled_module = runtime.modules.registry.get(file_instance)

    return CompiledDocument(
        resource_location=resource_location,
        ast=ast,
        diagnostics=[*errors, *compilation_unit.diagnostics.exceptions],
        compiled_unit=compilation_unit,
        compiled_module=compiled_module,
        ctx=ctx,
        dependents=set(),
    )


@contextmanager
def use_steps(mecha: Mecha, steps):
    initial_steps = mecha.steps
    mecha.steps = steps
    yield
    mecha.steps = initial_steps


async def compile(
    ctx: LanguageServerContext,
    resource_location: str,
    source_path: str,
    source_file: Function | Module,
) -> tuple[AstRoot, list[InvalidSyntax]]:
    mecha = ctx.inject(Mecha)
    diagnostics = []

    try:
        project_index = ctx.inject(AegisProjectIndex)
        project_index.remove_associated(source_path)
    except Exception as e:
        tb = "\n".join(traceback.format_tb(e.__traceback__))
        logging.error(f"{e}\n{tb}")

    indexer = Indexer(
        ctx=ctx,
        resource_location=resource_location,
        source_path=source_path,
        file_instance=source_file,
    )

    with use_steps(mecha, [indexer, mecha.lint, mecha.transform]):

        # Configure the database to compile the file
        compiled_unit = CompilationUnit(
            resource_location=resource_location, pack=ctx.data
        )
        database = mecha.database
        database[source_file] = compiled_unit
        database.enqueue(source_file)

        for step, file_instance in database.process_queue():
            compilation_unit = mecha.database[file_instance]
            logging.debug(f"--- Step {step} for {compilation_unit.filename} ---")
            start = time.time()

            if step < 0:
                try:
                    compilation_unit.source = file_instance.text
                    # Create the token stream
                    stream = TokenStream(
                        source=compilation_unit.source,
                        preprocessor=mecha.preprocessor,
                    )

                    ast = mecha.parse_stream(
                        mecha.spec.multiline, None, AstRoot.parser, stream  # type: ignore
                    )

                    ast, errors = ErrorAccumulator(
                        resource_location=resource_location,
                        filename=compilation_unit.filename,
                        file_instance=file_instance,
                    ).collect(ast)

                    diagnostics.extend(errors)

                    compilation_unit.ast = ast
                    mecha.database.enqueue(file_instance, 0)

                except InvalidSyntax as exec:
                    logging.error(f"Failed to parse: {exec}")
                except KeyError as exec:
                    tb = "\n".join(traceback.format_tb(exec.__traceback__))
                    logging.error(f"{tb}")
                except Exception as exec:
                    logging.error(f"{type(exec)}: {exec}")

            elif step < len(mecha.steps):
                if not compilation_unit.ast:
                    continue
                step_diagnostics = DiagnosticCollection()
                try:
                    with mecha.steps[step].use_diagnostics(step_diagnostics):
                        if ast := mecha.steps[step](compilation_unit.ast):
                            if not step_diagnostics.error:
                                compilation_unit.ast = ast
                                mecha.database.enqueue(
                                    key=file_instance,
                                    step=step + 1,
                                    priority=compilation_unit.priority,
                                )

                            compilation_unit.diagnostics.extend(step_diagnostics)
                except McCompilationError as e:
                    cause = e.__cause__
                    tb = traceback.extract_tb(cause.__traceback__)[-1]
                    logging.error(type(cause))
                    logging.error(tb)

                    if Path(tb.filename) == Path(source_path):
                        diagnostics.append(
                            Diagnostic(
                                message=str(cause),
                                level="error",
                                location=SourceLocation(
                                    0, tb.lineno or 0, tb.colno or 0
                                ),
                                end_location=SourceLocation(
                                    0, tb.end_lineno or 0, tb.end_colno or 0
                                ),
                            )
                        )

                    logging.error("\n".join(traceback.format_tb(cause.__traceback__)))

            logging.debug(f"Execution took {time.time() - start}s")
            
    return indexer.output_ast, diagnostics
