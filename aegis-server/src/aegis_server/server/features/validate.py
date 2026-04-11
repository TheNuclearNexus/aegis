import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from beet import (
    Advancement,
    Function,
    LootTable,
    Predicate,
    TextFileBase,
)
from beet.core.utils import extra_field, required_field
from bolt import Module, Runtime
from mecha import (
    AbstractNode,
    AstNode,
    AstRoot,
)
from mecha import CompilationError as McCompilationError
from mecha import (
    CompilationUnit,
    Diagnostic,
    DiagnosticCollection,
    Mecha,
    MutatingReducer,
    rule,
)
from mecha.ast import AstError
from mecha.contrib.nested_location import (
    NestedLocationResolver,
    NestedLocationTransformer,
)
from pygls.workspace import TextDocument
import lsprotocol.types as lsp
from tokenstream import InvalidSyntax, SourceLocation, TokenStream

from ..indexing import (
    AegisProjectIndex,
    AstFreezer,
    ProjectIndexer,
    TypeAnnotationResolver,
)
from ..shadows.compile_document import (
    COMPILATION_RESULTS,
    CompilationError,
    CompiledDocument,
)
from ..shadows.context import LanguageServerContext

TIMEOUT_DURATION = 30

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

        location, file = ctx.path_to_resource[path]

        if not isinstance(file, Function) and not isinstance(file, Module):
            COMPILATION_RESULTS[location] = CompiledDocument(
                ctx, location, None, [], None, None
            )
            logging.debug("File is not a function or module.")
            return []

        try:
            compiled_doc = await parse_function(
                ctx,
                location,
                text_doc.path,
                type(file)(text_doc.source, text_doc.path),
            )

            COMPILATION_RESULTS[location] = compiled_doc
            res = compiled_doc.diagnostics

        except TimeoutError as ex:
            message = (
                f"Compilation took longer than {TIMEOUT_DURATION} seconds, aborting"
            )
            logging.debug(f"{message}\n{ex}")
            ctx.ls.show_message(message, lsp.MessageType.Error)
            res = []

    return res



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
    mecha = ctx.inject(Mecha)
    runtime = ctx.inject(Runtime)

    start = time.time()

    loop = asyncio.get_running_loop()

    if file_instance in mecha.database:
        del mecha.database[file_instance]

    with ThreadPoolExecutor() as pool:
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    pool, compile, ctx, resource_location, source_path, file_instance
                ),
                timeout=TIMEOUT_DURATION,
            )
            ast, errors = results[file_instance]

        except TimeoutError as exec:
            raise exec

    logging.debug(f"Compilation for {source_path} took {time.time() - start}s")

    # # Parse the stream
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
    initial_steps = mecha.steps.copy()
    mecha.steps.clear()
    mecha.steps.extend(steps)
    yield
    mecha.steps.clear()
    mecha.steps.extend(initial_steps)


def compile(
    ctx: LanguageServerContext,
    resource_location: str,
    source_path: str,
    source_file: Function | Module,
) -> dict[TextFileBase[Any], tuple[AstNode, list[InvalidSyntax | Diagnostic]]]:
    mecha = ctx.inject(Mecha)
    runtime = ctx.inject(Runtime)
    diagnostics: dict[TextFileBase[Any], list[InvalidSyntax | Diagnostic]] = dict()

    try:
        project_index = ctx.inject(AegisProjectIndex)
        project_index.remove_associated(source_path)
    except Exception as e:
        tb = "\n".join(traceback.format_tb(e.__traceback__))
        logging.error(f"{e}\n{tb}")

    project_index[type(source_file)].add_definition(resource_location, source_path)

    logging.debug(f"{list(ctx.data.functions.keys())}")

    nested_location_transformer = NestedLocationTransformer(
        nested_location_resolver=NestedLocationResolver(ctx=ctx)
    )

    project_indexer = ProjectIndexer(
        ctx=ctx,
        index=project_index,
        mecha=mecha,
        # argument parser to resource type
        parser_to_file_type={
            "minecraft:advancement": Advancement,
            "minecraft:function": Function,
            "minecraft:predicate": Predicate,
            "minecraft:loot_table": LootTable,
        },
    )

    try:
        type_annotation_resolver = TypeAnnotationResolver(
            ctx=ctx,
            index=project_index,
            runtime=runtime,
            mecha=mecha,
        )
    except Exception as e:
        tb = "\n".join(traceback.format_tb(e.__traceback__))
        logging.error(f"{e}\n{tb}")
        raise e

    frozen_asts = AstFreezer(mecha=mecha)

    with use_steps(
        mecha,
        [
            nested_location_transformer,
            project_indexer,
            type_annotation_resolver,
            frozen_asts,
            mecha.lint,
            mecha.transform,
        ],
    ):
        # Configure the database to compile the file
        compiled_unit = CompilationUnit(
            resource_location=resource_location, pack=ctx.data
        )
        database = mecha.database
        database[source_file] = compiled_unit
        database.enqueue(source_file)

        for step, file_instance in database.process_queue():
            compilation_unit = mecha.database.get(file_instance)

            if not compilation_unit:
                logging.debug(
                    f"--- Step {step} could not find a compilation unit for {file_instance} ---"
                )
                continue

            logging.debug(
                f"--- Step {step} for {compilation_unit.filename} @ {compilation_unit.priority} ---"
            )
            start = time.time()

            if step < 0:
                if compilation_unit.ast is not None:
                    mecha.database.enqueue(file_instance, 0, compilation_unit.priority)
                    continue

                try:
                    compilation_unit.source = file_instance.text
                    # Create the token stream
                    stream = TokenStream(
                        source=compilation_unit.source,
                        preprocessor=mecha.preprocessor,
                    )

                    ast = mecha.parse_stream(
                        mecha.spec.multiline,
                        None,
                        AstRoot.parser or "root",
                        stream,  # type: ignore
                    )

                    ast, errors = ErrorAccumulator(
                        resource_location=resource_location,
                        filename=compilation_unit.filename,
                        file_instance=file_instance,
                    ).collect(ast)

                    diagnostics.setdefault(file_instance, []).extend(errors)

                    compilation_unit.ast = ast
                    mecha.database.enqueue(file_instance, 0, compilation_unit.priority)

                except InvalidSyntax as exec:
                    logging.error(f"Failed to parse: {exec}")
                except KeyError as exec:
                    tb = "\n".join(traceback.format_tb(exec.__traceback__))
                    logging.error(f"{tb}")
                except Exception as exec:
                    tb = "\n".join(traceback.format_tb(exec.__traceback__))
                    logging.error(f"{type(exec)}: {exec}\n{tb}")

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
                        else:
                            logging.debug(f"Step {step} yielded no ast?")
                except McCompilationError as e:
                    cause = e.__cause__
                    tb = traceback.extract_tb(cause.__traceback__)[-1]
                    logging.error(type(cause))
                    logging.error(tb)

                    if Path(tb.filename) == Path(source_path):
                        diagnostics.setdefault(file_instance, []).append(
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

    results: dict[
        TextFileBase[Any], tuple[AstNode, list[InvalidSyntax | Diagnostic]]
    ] = dict()
    for file in frozen_asts:
        results[file] = (
            frozen_asts[file],
            diagnostics.get(file, []),
        )

    return results
