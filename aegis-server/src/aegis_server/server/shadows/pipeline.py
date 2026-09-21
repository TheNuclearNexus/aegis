from typing import cast

import lsprotocol.types as lsp
import traceback as tb
from beet import Context, GenericPlugin, Pipeline, Task

from .context import LanguageServerContext

__all__ = ["PipelineShadow"]


class PipelineShadow(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep a reference to every plugin we advance but never finish. Dropping
        # the returned Task lets the underlying generator be garbage collected
        # immediately, which runs its ``finally`` block while the object that
        # registered it is still being initialized. Bolt's ``Runtime.finalize``
        # is the canonical example: it is registered before ``Runtime.memo`` is
        # assigned, so early finalization raises
        # ``AttributeError: 'Runtime' object has no attribute 'memo'``.
        self.deferred_tasks: list[Task] = []

    def require(self, *args: GenericPlugin[Context] | str):
        for spec in args:
            try:
                plugin = self.resolve(spec)
            except Exception as exc:
                ls = cast(LanguageServerContext, self.ctx).ls
                traceback = '\n'.join(tb.format_tb(exc.__traceback__))
                message = f"An issue occured while loading plugin: {spec}\n{exc}\n{traceback}"
                ls.show_message(message.split("\n")[0], lsp.MessageType.Warning)
                ls.show_message_log(message, lsp.MessageType.Warning)
                continue

            if plugin in self.plugins:
                continue

            self.plugins.add(plugin)

            try:
                # Advance the plugin only once, ignore remaining work
                # Most setup happens in the first half of the plugin
                # where side effects happen in the latter
                if task := Task(plugin).advance(self.ctx):
                    self.deferred_tasks.append(task)
            except Exception as exc:
                ls = cast(LanguageServerContext, self.ctx).ls
                traceback = '\n'.join(tb.format_tb(exc.__traceback__))
                message = f"An issue occured while running first step of plugin: {plugin}\n{exc}\n{traceback}"
                ls.show_message(message.split("\n")[0], lsp.MessageType.Warning)
                ls.show_message_log(message, lsp.MessageType.Warning)
