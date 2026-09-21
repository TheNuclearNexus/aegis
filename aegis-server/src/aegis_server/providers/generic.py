import logging
import re
from typing import ClassVar
from aegis_core.ast.features.provider import BaseFeatureProvider
from aegis_core.ast.helpers import offset_location
from aegis_core.semantics import IMPLICIT_PARAMETERS, TokenModifier, TokenType
from bolt import (
    AstClassName,
    AstFormatString,
    AstFunctionSignature,
    AstFunctionSignatureArgument,
    AstFunctionSignatureVariadicArgument,
    AstFunctionSignatureVariadicKeywordArgument,
    AstValue,
)
from mecha import AstNode
from mecha.ast import AstItemSlot
from tokenstream import SourceLocation


__all__ = [
    "ItemSlotProvider",
    "ClassNameProvider",
    "FunctionSignatureProvider",
    "ValueProvider",
    "FormatStringProvider",
]


# --- Mecha ---
class ItemSlotProvider(BaseFeatureProvider[AstItemSlot]):
    @classmethod
    def semantics(cls, params):
        return [(params.node, "variable", ["readonly"])]


# --- Bolt ---
class ClassNameProvider(BaseFeatureProvider[AstClassName]):
    @classmethod
    def semantics(cls, params):
        return [(params.node, "class", [])]


def _name_node(node: AstNode, name: str):
    return AstNode(node.location, offset_location(node.location, len(name)))


class FunctionSignatureProvider(BaseFeatureProvider[AstFunctionSignature]):
    @classmethod
    def semantics(cls, params):
        signature = params.node
        name = signature.name
        magic = name.startswith("__") and name.endswith("__")

        tokens: list[tuple[AstNode, TokenType, list[TokenModifier]]] = [
            (_name_node(signature, name), "magicFunction" if magic else "function", [])
        ]

        for index, argument in enumerate(signature.arguments):
            match argument:
                case AstFunctionSignatureArgument():
                    kind = "parameter"
                    if index == 0:
                        kind = IMPLICIT_PARAMETERS.get(argument.name, kind)

                    tokens.append((_name_node(argument, argument.name), kind, []))
                case (
                    AstFunctionSignatureVariadicArgument()
                    | AstFunctionSignatureVariadicKeywordArgument()
                ):
                    end = argument.end_location
                    start = offset_location(end, -len(argument.name))
                    tokens.append((AstNode(start, end), "parameter", []))

        return tokens


class ValueProvider(BaseFeatureProvider[AstValue]):
    @classmethod
    def semantics(cls, params):
        match params.node.value:
            case bool() | None:
                return [(params.node, "builtinConstant", [])]
            case int() | float():
                return [(params.node, "number", [])]
            case str():
                return [(params.node, "string", [])]

        return None


class FormatStringProvider(BaseFeatureProvider[AstFormatString]):
    FORMAT_REGEX: ClassVar[re.Pattern] = re.compile(r"\{(:.+)?\}")

    @classmethod
    def semantics(cls, params):
        format_string = params.node

        formats = cls.FORMAT_REGEX.findall(format_string.fmt)

        nodes: list[tuple[AstNode, TokenType, list[TokenModifier]]] = []

        nodes.append(
            (
                AstNode(
                    format_string.location, offset_location(format_string.location, 1)
                ),
                "macro",
                [],
            )
        )

        for format, value in zip(formats, format_string.values):
            nodes.append(
                (
                    AstNode(offset_location(value.location, -1), value.location),
                    "macro",
                    [],
                )
            )
            nodes.append(
                (
                    AstNode(
                        value.end_location,
                        offset_location(value.end_location, 1 + len(format)),
                    ),
                    "macro",
                    [],
                )
            )

        return nodes
