from collections.abc import Iterator
import logging
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path

from aegis_server.providers import register_providers
from beet import (
    LATEST_MINECRAFT_VERSION,
    Context,
    PluginSpec,
    Project,
    ProjectBuilder,
    ProjectConfig,
    TemplateManager,
)
from beet.contrib.load import load
from beet.core.utils import change_directory, normalize_string
from mecha import Mecha
from pygls.server import LanguageServer

from ..indexing import AegisProjectIndex
from ..patches import apply_patches
from .context import LanguageServerContext, get_excluded_plugins
from .pipeline import PipelineShadow

__all__ = ["ProjectBuilderShadow"]


class ProjectBuilderShadow(ProjectBuilder):
    ls: LanguageServer

    def __init__(
        self,
        ls: LanguageServer,
        project: Project,
        root: bool = False,
        tmpdir: bool = False,
    ):
        self.ls = ls
        super().__init__(project, root, tmpdir)

    def bootstrap(self, ctx: Context):
        """Plugin that handles the project configuration."""

        excluded_plugins = get_excluded_plugins(ctx)
        plugins = self.config.require

        for plugin in plugins:
            if plugin in excluded_plugins:
                continue
            logging.debug(f"Requiring plugin {plugin}")

            ctx.require(plugin)

    # This stripped down version of build only handles loading the plugins from config
    # all other operations are gone such as linking
    @contextmanager
    def build(self) -> Iterator[LanguageServerContext]:
        """Create the context, run the pipeline, and return the context."""
        with ExitStack() as stack:
            name = self.config.name or self.project.directory.stem
            meta = deepcopy(self.config.meta)

            tmpdir = None
            cache = self.project.cache

            logging.debug("Creating context...")
            ctx = LanguageServerContext(
                _pipeline=PipelineShadow,
                ls=self.ls,
                project_config=self.config,
                project_id=self.config.id or normalize_string(name),
                project_name=name,
                project_description=self.config.description,
                project_author=self.config.author,
                project_version=self.config.version,
                project_root=self.root,
                minecraft_version=(
                    self.config.minecraft
                    if len(self.config.minecraft) > 0
                    else LATEST_MINECRAFT_VERSION
                ),
                directory=self.project.directory,
                output_directory=self.project.output_directory,
                meta=meta,
                cache=cache,
                worker=stack.enter_context(self.project.worker_pool.handle()),
                template=TemplateManager(
                    templates=self.project.template_directories,
                    cache_dir=cache["template"].directory,
                ),
                whitelist=self.config.whitelist,
            )

            pipelined_plugins: list[PluginSpec] = [self.bootstrap]

            excluded_plugins = get_excluded_plugins(ctx)
            for item in self.config.pipeline:
                if isinstance(item, str) and (
                    item == "mecha" or item in excluded_plugins
                ):
                    continue

                if isinstance(item, ProjectConfig):
                    pipelined_plugins.append(
                        ProjectBuilderShadow(
                            self.ls,
                            Project(
                                resolved_cache=ctx.cache,
                                resolved_config=item,
                                resolved_worker_pool=self.project.worker_pool,
                            ),
                        )
                    )
                else:
                    pipelined_plugins.append(item)

            logging.debug("Configuring Context")
            configure_ctx(ctx)

            with change_directory(tmpdir):
                for plugin in pipelined_plugins:
                    logging.debug(f"Running pipeline {plugin}")
                    ctx.require(plugin)
                # pipeline = stack.enter_context(ctx.activate())
                # pipeline.run(plugins)

            # Load everything into context *after* the first half of the plugins
            # are ran by the pipeline
            logging.debug("Loading assets")
            load(
                resource_pack=self.config.resource_pack.load,
                data_pack=self.config.data_pack.load,
            )(ctx)

            project_index = ctx.inject(AegisProjectIndex)

            mc = ctx.inject(Mecha)

            for pack in ctx.packs:
                logging.debug("Enqueuing files in database")
                # Add file to the compilation database
                for provider in mc.providers:
                    for file_instance, compilation_unit in provider(pack, mc.match):
                        mc.database[file_instance] = compilation_unit
                        # mc.database.enqueue(file_instance)

                logging.debug("Adding all files to index")
                # Build a map of file path to resource location
                for location, file in pack.all():
                    try:
                        path = Path(file.ensure_source_path())
                        ctx.path_to_resource[path] = (location, file)
                        project_index[type(file)].add_definition(location, path)
                    except:
                        continue

            yield ctx

    def __call__(self, ctx: Context):
        """The builder instance is itself a plugin used for merging subpipelines."""
        with self.build() as child_ctx:
            if isinstance(ctx, LanguageServerContext):
                ctx.children.append(child_ctx)


def configure_ctx(ctx: LanguageServerContext):
    # mc.steps = mc.steps[: mc.steps.index(mc.lint) + 1]

    register_providers(ctx)
    apply_patches()
