from dataclasses import dataclass, field
from typing import Generic, TypeVar

import lsprotocol.types as lsp
from beet import Context
from mecha import AstNode

from ...semantics import TokenModifier, TokenType

__all__ = [
    "BaseFeatureProvider",
    "BaseParams",
    "CompletionParams",
    "HoverParams",
    "DefinitionParams",
    "ReferencesParams",
    "RenameParams",
    "SemanticsParams",
]

Node = TypeVar("Node", bound=AstNode)


@dataclass
class BaseParams(Generic[Node]):
    ctx: Context
    node: Node
    resource_location: str


@dataclass
class CompletionParams(BaseParams[Node]): ...


@dataclass
class HoverParams(BaseParams[Node]):
    text_range: lsp.Range


@dataclass
class DefinitionParams(BaseParams[Node]): ...


@dataclass
class ReferencesParams(BaseParams[Node]): ...


@dataclass
class RenameParams(BaseParams[Node]): ...


@dataclass
class SemanticsParams(BaseParams[Node]): ...


class BaseFeatureProvider(Generic[Node]):

    @classmethod
    def completion(
        cls, params: CompletionParams[Node]
    ) -> lsp.CompletionList | None:
        return None

    @classmethod
    def hover(cls, params: HoverParams[Node]) -> lsp.Hover | None:
        return None

    @classmethod
    def definition(
        cls, params: DefinitionParams[Node]
    ) -> list[lsp.Location | lsp.LocationLink] | lsp.Location | lsp.LocationLink | None:
        return None

    @classmethod
    def references(cls, params: ReferencesParams[Node]) -> list[lsp.Location] | None:
        return None

    @classmethod
    def rename(cls, params: RenameParams[Node]) -> lsp.WorkspaceEdit | None:
        return None

    @classmethod
    def semantics(
        cls, params: SemanticsParams[Node]
    ) -> list[tuple[AstNode, TokenType, list[TokenModifier]]] | None:
        return None
