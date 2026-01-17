from dataclasses import dataclass
from typing import Any

from beet.core.utils import extra_field
from bolt import CompiledModule
from mecha import AstNode, CompilationUnit, Diagnostic
from tokenstream import InvalidSyntax

from .context import LanguageServerContext

__all__ = ["CompiledDocument"]


COMPILATION_RESULTS: dict[str, "CompiledDocument"] = {}

CompilationError = InvalidSyntax | Diagnostic


@dataclass
class CompiledDocument:
    ctx: LanguageServerContext

    resource_location: str

    ast: AstNode | None
    diagnostics: list[CompilationError]

    compiled_unit: CompilationUnit | None
    compiled_module: CompiledModule | None

    dependents: set[str] = extra_field(default_factory=set)
