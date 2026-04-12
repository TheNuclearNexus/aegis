import asyncio
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Thread
import traceback
from typing import Generator, cast
from urllib import request
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from beet import (
    Context,
    DataPack,
    Function,
    PluginError,
    PluginImportError,
    Project,
    ProjectConfig,
    load_config,
    locate_config,
)
from beet.core.utils import change_directory
from beet.library.base import LATEST_MINECRAFT_VERSION
from bolt import Module
from lsprotocol import types as lsp
from mecha import DiagnosticErrorSummary
from pygls.server import LanguageServer
from pygls.workspace import TextDocument

from aegis_core.registry import AegisGameRegistries

from .features.validate import validate_function
from .shadows.compile_document import COMPILATION_RESULTS
from .shadows.context import LanguageServerContext
from .shadows.project_builder import ProjectBuilderShadow

logging.basicConfig(
    filename="aegis.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(levelname)s:%(filename)s:%(lineno)d:\t%(message)s",
)

SUPPORTED_EXTENSIONS = [Function.extension, Module.extension]

def get_parent_context(ctx: LanguageServerContext, file_path: Path) -> LanguageServerContext | None:
    """Try to mount a given file path to the context. True if the file was successfully mounted"""

    if file_path.suffix not in SUPPORTED_EXTENSIONS:
        return None
    
    if file_path in ctx.path_to_resource:
        return ctx
    
    for child in ctx.children:
        if parent := get_parent_context(child, file_path):
            return parent

    load_options = ctx.project_config.data_pack.load
    prefix = None
    for entry in load_options.entries():
        if isinstance(entry, dict):
            for key, paths in entry.items():
                for mount_path in paths.entries():
                    if file_path.is_relative_to(mount_path):
                        relative = file_path.relative_to(mount_path)
                        prefix = str(key / relative)
                        break

        elif file_path.is_relative_to(entry):
            relative = file_path.relative_to(entry)
            prefix = str(relative)

    if prefix is None:
        return None

    try:
        # Mounting is done into a temp datapack to make it easier to get the file path
        # The other option which may be better is to monkey patch mount directly
        temp = DataPack()
        temp.mount(prefix, file_path)
        for [location, file] in temp.all():
            if not (isinstance(file, Function) or isinstance(file, Module)):
                continue

            mount_path = Path(file.ensure_source_path())
            ctx.path_to_resource[mount_path] = (location, file)
            ctx.data[type(file)][location] = file

            logging.debug(f"Mounted {file_path} to {location}")
        return ctx
    except Exception as exc:
        logging.error(f"Failed to mount {file_path}, reloading datapack,\n{exc}")

    return None



class AegisServer(LanguageServer):
    _instances: dict[Path, tuple[Lock, LanguageServerContext]] = dict()
    _sites: list[str] = []
    _index_thread: Thread
    _alive: bool = True

    def set_sites(self, sites: list[str]):
        self._sites = sites

    def __init__(self, *args):
        super().__init__(*args)
        self._instances = {}
        self._index_thread = Thread(
            target=lambda self, parent: self.scan_functions(parent),
            args=[self, threading.current_thread()],
        )
        # self._index_thread.start()

    async def index_functions(self, ctx: LanguageServerContext):
        for function, file in cast(
            list[tuple[str, Function | Module]],
            [*ctx.data.functions.items(), *ctx.data[Module].items()],
        ):
            if function not in COMPILATION_RESULTS and file.source_path:
                self.show_message_log("indexing " + function, lsp.MessageType.Debug)
                await validate_function(
                    ctx, self.workspace.get_document(Path(file.source_path).as_uri())
                )
                break

    def scan_functions(self, parent_thread: Thread):
        logging.info("Started Indexing Thread")
        try:
            while self._alive and parent_thread.is_alive():
                for lock, ctx in self._instances.values():
                    if not lock.acquire(blocking=False):
                        continue

                    asyncio.run(self.index_functions(ctx))

                    lock.release()
                    time.sleep(0.1)
        except Exception as exc:
            lock.release()
            tb = "\n".join(traceback.format_tb(exc.__traceback__))
            logging.error(f"Fatal error occured while indexing function!\n{exc}\n{tb}")

        logging.info("Stopped Indexing Thread")

    def load_registry(self, ctx: Context, minecraft_version: str):
        """Load the game registry from Misode's mcmeta repository"""

        if len(minecraft_version) <= 0:
            minecraft_version = LATEST_MINECRAFT_VERSION

        cache_dir = Path("./.aegis_cache")
        if not cache_dir.exists():
            os.mkdir(cache_dir)

        if not (cache_dir / "registries").exists():
            os.mkdir(cache_dir / "registries")

        file_path = cache_dir / "registries" / (minecraft_version + ".json")

        # logging.debug(minecraft_version)

        if not file_path.exists():
            try:
                request.urlretrieve(
                    f"https://raw.githubusercontent.com/misode/mcmeta/refs/tags/{minecraft_version}-summary/registries/data.min.json",
                    file_path,
                )
            except Exception as exc:
                self.show_message(
                    f"Failed to download registry for version {minecraft_version}, completions will be disabled\n{exc}",
                    lsp.MessageType.Error,
                )

                return

        try:
            with open(file_path, "r") as file:
                try:
                    registries = json.loads(file.read())

                    ctx.inject(AegisGameRegistries).registries = registries

                except json.JSONDecodeError as exc:
                    self.show_message(
                        f"Failed to parse registry for version {minecraft_version}, completions will be disabled\n{exc}",
                        lsp.MessageType.Error,
                    )
                    os.remove(file_path)

        except Exception as exc:
            self.show_message(
                f"An unhandled exception occured loading registry for version {minecraft_version}\n{exc}",
                lsp.MessageType.Error,
            )
            os.remove(file_path)

    def create_context(
        self, config: ProjectConfig, config_path: Path
    ) -> LanguageServerContext:
        """Attempt to configure the project's context and run necessary plugins"""
        project = Project(config, None, config_path)

        with ProjectBuilderShadow(self, project, root=True).build() as ctx:
            return ctx

    def create_instance(
        self, config: ProjectConfig, config_path: Path
    ) -> LanguageServerContext | None:
        # Ensure that we aren't loading in all project files
        config.output = None

        og_sys_path = sys.path
        og_modules = sys.modules

        sys.path = [*self._sites, str(config_path.parent), *og_sys_path]

        with change_directory(config_path.parent):
            instance = None

            try:
                instance = self.create_context(config, config_path)
            except PluginImportError as plugin_error:
                logging.error(
                    f"Plugin Import Error: {plugin_error}\n{plugin_error.__cause__}"
                )
            except PluginError as plugin_error:
                logging.error(plugin_error.__cause__)
                raise plugin_error.__cause__
            except DiagnosticErrorSummary as summary:
                logging.error("Errors found in the following:")
                for diag in summary.diagnostics.exceptions:
                    logging.error(
                        "\t" + str(diag.file.source_path if diag.file is not None else "")
                    )

            except Exception as e:
                logging.error(f"Error occured while running beet: {type(e)} {e}")

        sys.path = og_sys_path
        sys.modules = og_modules

        if instance:
            self.load_registry(instance, config.minecraft)

        return instance

    def setup_workspaces(self):
        config_paths: list[Path] = []

        for w in self.workspace.folders.values():
            ws_path = self.uri_to_path(w.uri)

            if path := locate_config(ws_path):
                config_paths.append(path)

        for config_path in config_paths:
            try:
                config = load_config(config_path)
                if config := self.create_instance(config, config_path):
                    self._instances[config_path.parent] = (Lock(), config)
            except Exception as exc:
                logging.error(
                    f"Failed to load config at {config_path} due to the following\n{exc}"
                )

    def uri_to_path(self, uri: str):
        parsed = urlparse(uri)
        host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
        norm_path = os.path.normpath(
            os.path.join(host, url2pathname(unquote(parsed.path)))
        )
        # # logging.debug(norm_path)
        norm_path = Path(norm_path)
        return norm_path

    def get_instance(self, config_path: Path):
        if config_path not in self._instances or self._instances[config_path] is None:
            config = load_config(config_path)
            instance = self.create_instance(config, config_path)

            if instance is not None:
                self._instances[config_path] = (Lock(), instance)

        return self._instances[config_path]

    @contextmanager
    def context(
        self, document: TextDocument
    ) -> Generator[LanguageServerContext | None, None, None]:
        doc_path = Path(document.path).resolve()

        parents: list[Path] = []

        for parent_path in self._instances.keys():
            if doc_path.resolve().is_relative_to(parent_path):
                parents.append(parent_path)

        if len(parents) == 0:
            yield None
            return

        (lock, context) = self.get_instance(parents[-1])

        context = get_parent_context(context, doc_path)

        if context is None:
            yield None
            return

        lock.acquire()
        yield context
        lock.release()

    def _kill(self):
        self._alive = False

    def shutdown(self):
        self._kill()
        super().shutdown()
