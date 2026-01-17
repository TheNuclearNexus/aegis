import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import lsprotocol.types as lsp
from beet import Context, NamespaceFile, PluginError, PluginSpec, ProjectConfig
from beet.core.utils import extra_field, local_import_path, required_field
from pygls.server import LanguageServer

__all__ = ["LanguageServerContext"]


# We use this shadow of context in order to route calls to `ctx`
# to our own methods, this allows us to bypass side effects without
# having to break plugins
@dataclass(frozen=True)
class LanguageServerContext(Context):
    ls: LanguageServer = required_field()
    project_config: ProjectConfig = required_field()
    _pipeline: type = required_field()

    project_uuid: str = extra_field(default_factory=lambda: str(uuid.uuid1()))

    path_to_resource: dict[str, tuple[str, NamespaceFile]] = extra_field(
        default_factory=dict
    )

    def require(self, *args: PluginSpec):
        """Execute the specified plugin."""
        for arg in args:
            try:
                self.inject(self._pipeline).require(arg)
            except PluginError as exc:
                message = f"Failed to load plugin: {arg}\n{exc}"
                self.ls.show_message(message.split("\n")[0], lsp.MessageType.Error)
                self.ls.show_message_log(message, lsp.MessageType.Error)

    @contextmanager
    def activate(self):
        """Push the context directory to sys.path and handle cleanup to allow module reloading."""
        with local_import_path(str(self.directory.resolve())), self.cache:
            yield self.inject(self._pipeline)

    def get_resource_from_path(self, path: str) -> tuple[str, NamespaceFile] | None:
        return self.path_to_resource.get(path)


def get_excluded_plugins(ctx: Context):
    lsp_config: dict = ctx.meta.setdefault("lsp", {})

    excluded_plugins = lsp_config.get("excluded_plugins") or []

    return excluded_plugins
