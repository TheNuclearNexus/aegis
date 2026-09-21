import builtins, logging, traceback
from dataclasses import dataclass, field
from typing import get_args

from aegis_core.ast.features import AegisFeatureProviders
from aegis_core.ast.features.provider import SemanticsParams
from aegis_core.ast.helpers import offset_location
from aegis_core.ast.metadata import ResourceLocationMetadata, retrieve_metadata
from beet import Context
from beet.core.utils import required_field
from bolt import (
    AstAttribute,
    AstCall,
    AstDecorator,
    AstFromImport,
    AstIdentifier,
    AstImportedItem,
    AstPrelude,
)
from lsprotocol import types as lsp
from mecha import (
    AstCommand,
    AstNode,
    AstResourceLocation,
    Mecha,
    Reducer,
    rule,
)
from tokenstream import SourceLocation

from aegis_core.semantics import PASCAL_CASE, TokenModifier, TokenType

from ...server import AegisServer
from ..features.validate import get_compilation_data

BUILTIN_CLASSES = {
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and not name.startswith("_")
}

TOKEN_TYPES: dict[TokenType, int] = {
    get_args(literal)[0]: i for (i, literal) in enumerate(get_args(TokenType))
}

TOKEN_MODIFIERS: dict[TokenModifier, int] = {
    get_args(literal)[0]: pow(2, i)
    for (i, literal) in enumerate(get_args(TokenModifier))
}

# logging.debug(TOKEN_MODIFIERS)
# token tuple
# 0: line offset
# 1: col offset
# 2: length
# 3: type
# 4: modifier bitflag


def node_to_token(
    node: AstNode,
    type: int,
    modifier: int,
    prev_node: AstNode | None,
) -> tuple[int, ...]:
    line_offset = node.location.lineno - 1
    col_offset = node.location.colno - 1
    length = node.end_location.pos - node.location.pos

    if prev_node is not None:
        line_offset -= prev_node.location.lineno - 1

        if line_offset == 0:
            col_offset -= prev_node.location.colno - 1

    token = (line_offset, col_offset, length, type, modifier)
    return token


@dataclass
class SemanticTokenCollector(Reducer):
    nodes: list[tuple[AstNode, int, int]] = field(default_factory=list)
    overrides: list[tuple[AstNode, int, int]] = field(default_factory=list)
    imported_modules: set[str] = field(default_factory=set)
    ctx: Context = required_field()
    resource_location: str = required_field()
    source: str = required_field()

    @rule(AstCommand)
    def command(self, node: AstCommand):
        match node.identifier:
            case "import:module":
                modules: list[AstResourceLocation] = node.arguments  # type: ignore

                for m in modules:
                    self.nodes.append(
                        (
                            m,
                            TOKEN_TYPES["module"],
                            0,
                        )
                    )
                    self.imported_modules.add(m.path.partition(".")[0])
            case "import:module:as:alias":
                module: AstResourceLocation = node.arguments[0]  # type: ignore
                item: AstImportedItem = node.arguments[1]  # type: ignore

                type = TOKEN_TYPES["module"]

                self.nodes.append((module, type, 0))
                self.nodes.append((item, type, 0))
                self.imported_modules.add(item.name)

        end_location = node.end_location

        prototypes = self.ctx.inject(Mecha).spec.prototypes

        if len(node.arguments) > 0:
            if node.identifier in prototypes:
                name_length = len(prototypes[node.identifier].signature[0])

                end_location = SourceLocation(
                    lineno=node.location.lineno,
                    pos=node.location.pos + name_length,
                    colno=node.location.colno + name_length,
                )
            else:
                return

        temp_node = AstNode(location=node.location, end_location=end_location)

        self.nodes.append(
            (
                temp_node,
                (
                    TOKEN_TYPES["keyword"]
                    if "subcommand" not in node.identifier
                    or node.identifier == "execute:subcommand"
                    else TOKEN_TYPES["macro"]
                ),
                0,
            )
        )

    @rule(AstFromImport)
    def from_import(self, from_import: AstFromImport):
        if isinstance(from_import, AstPrelude):
            return

        # logging.debug(from_import)

        location: AstResourceLocation = from_import.arguments[0]  # type: ignore
        imports: tuple[AstImportedItem] = from_import.arguments[1:]  # type: ignore

        self.nodes.append(
            (
                location,
                TOKEN_TYPES["module"],
                0,
            )
        )

        import_offset = len("import")
        self.nodes.append(
            (
                AstNode(
                    offset_location(location.end_location, 1),
                    offset_location(location.end_location, import_offset + 1),
                ),
                TOKEN_TYPES["keyword"],
                0,
            )
        )

    def override_callable(self, target: AstNode):
        if isinstance(target, AstIdentifier):
            name, location = target.value, target.location
            builtin = name in BUILTIN_CLASSES
        elif isinstance(target, AstAttribute):
            name = target.name
            location = offset_location(target.end_location, -len(name))
            builtin = False
        else:
            return

        self.overrides.append(
            (
                AstNode(location, offset_location(location, len(name))),
                TOKEN_TYPES[
                    "class" if builtin or PASCAL_CASE.match(name) else "function"
                ],
                0,
            )
        )

    @rule(AstCall)
    def call(self, node: AstCall):
        self.override_callable(node.value)

    @rule(AstDecorator)
    def decorator(self, node: AstDecorator):
        self.nodes.append(
            (
                AstNode(node.location, offset_location(node.location, 1)),
                TOKEN_TYPES["decorator"],
                0,
            )
        )

        target = node.expression
        if isinstance(target, AstCall):
            target = target.value

        while isinstance(target, AstAttribute):
            location = offset_location(target.end_location, -len(target.name))
            self.overrides.append(
                (AstNode(location, target.end_location), TOKEN_TYPES["decorator"], 0)
            )
            target = target.value

        if (
            isinstance(target, AstIdentifier)
            and target.value not in self.imported_modules
        ):
            self.overrides.append((target, TOKEN_TYPES["decorator"], 0))

    @rule(AstIdentifier)
    def identifier(self, node: AstIdentifier):
        if node.value in self.imported_modules:
            self.overrides.append((node, TOKEN_TYPES["module"], 0))

    @rule(AstNode)
    def node(self, node: AstNode):
        provider = self.ctx.inject(AegisFeatureProviders).retrieve(node)
        try:
            if isinstance(node, AstResourceLocation):
                metadata = retrieve_metadata(
                    self.resource_location, node, ResourceLocationMetadata
                )
                if self.source[node.location.pos : node.end_location.pos] not in (
                    node.get_value(),
                    metadata.unresolved_path if metadata else None,
                ):
                    return
            
            tokens = provider.semantics(
                SemanticsParams(self.ctx, node, self.resource_location)
            )

            if not tokens or len(tokens) == 0:
                return

            self.nodes.extend(
                (
                    node,
                    TOKEN_TYPES[token_type],
                    sum(map(lambda m: TOKEN_MODIFIERS[m], token_modifiers)),
                )
                for node, token_type, token_modifiers in tokens
            )
        except Exception as e:
            tb = '\n'.join(traceback.format_tb(e.__traceback__))
            logging.error(
                f"An error occured running provider {provider}\n{e}\n{tb}"
            )


    def walk(self, root: AstNode):
        self.nodes = []
        self.overrides = []
        self.imported_modules = set()
        self.__call__(root)
        self.nodes.extend(self.overrides)

        tokens: list[tuple[int, ...]] = []

        self.nodes = list(
            {
                (n.location.pos, n.end_location.pos): (n, t, m)
                for n, t, m in self.nodes
            }.values()
        )

        self.nodes.sort(key=lambda n: (n[0].location.lineno, n[0].location.colno))

        prev_node = None
        for node, type, modifier in self.nodes:
            if node.end_location.pos - node.location.pos <= 0:
                continue

            tokens.append(node_to_token(node, type, modifier, prev_node))
            prev_node = node

        return list(sum(tokens, ()))


async def semantic_tokens(ls: AegisServer, params: lsp.SemanticTokensParams):
    text_doc = ls.workspace.get_document(params.text_document.uri)
    with ls.context(text_doc) as ctx:
        if ctx is None:
            data = []
        else:
            if compiled_doc := await get_compilation_data(ctx, text_doc):
                ast = compiled_doc.ast

                data = (
                    SemanticTokenCollector(
                        ctx=ctx,
                        resource_location=compiled_doc.resource_location,
                        source=text_doc.source,
                    ).walk(ast)
                    if ast
                    else []
                )
            else:
                data = []

    return lsp.SemanticTokens(data=data or [])
