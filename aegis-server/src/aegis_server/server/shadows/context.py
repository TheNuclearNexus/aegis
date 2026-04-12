from pathlib import Path
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import lsprotocol.types as lsp
from beet import Context, NamespaceFile, PluginError, PluginSpec, ProjectConfig
from beet.core.utils import extra_field, local_import_path, required_field
from pygls.server import LanguageServer

__all__ = ["LanguageServerContext"]


class PathToResource(dict[Path, tuple[str, NamespaceFile]]):
    def __init__(self):
        super().__init__()

    def __getitem__(self, key: str | Path) -> tuple[str, NamespaceFile]:
        if isinstance(key, str):
            key = Path(key)

        key = key.resolve()
        return super().__getitem__(key)

    def __setitem__(self, key: str | Path, value: tuple[str, NamespaceFile]) -> None:
        if isinstance(key, str):
            key = Path(key)

        key = key.resolve()
        super().__setitem__(key, value)

    def get(
        self, key: Path | str, default: tuple[str, NamespaceFile] | None = None
    ) -> tuple[str, NamespaceFile] | None:
        if isinstance(key, str):
            key = Path(key)

        key = key.resolve()
        return super().get(key, default)

    def setdefault(
        self, key: str | Path, default: tuple[str, NamespaceFile]
    ) -> tuple[str, NamespaceFile]:
        if isinstance(key, str):
            key = Path(key)

        key = key.resolve()
        return super().setdefault(key, default)


# We use this shadow of context in order to route calls to `ctx`
# to our own methods, this allows us to bypass side effects without
# having to break plugins
@dataclass(frozen=True)
class LanguageServerContext(Context):
    ls: LanguageServer = required_field()
    project_config: ProjectConfig = required_field()
    _pipeline: type = required_field()

    project_uuid: str = extra_field(default_factory=lambda: str(uuid.uuid1()))

    path_to_resource: PathToResource = extra_field(default_factory=PathToResource)

    children: list["LanguageServerContext"] = extra_field(default_factory=list)

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
