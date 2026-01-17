import logging
import re
from types import NoneType
from typing import ClassVar
from aegis_core.ast.features.provider import BaseFeatureProvider
from aegis_core.ast.helpers import offset_location
from aegis_core.ast.metadata import VariableMetadata, retrieve_metadata
from aegis_core.semantics import TokenModifier, TokenType
from bolt import (
    AstClassName,
    AstFormatString,
    AstFunctionSignature,
    AstFunctionSignatureArgument,
    AstValue,
)
from mecha import AstNode
from mecha.ast import AstItemSlot
from tokenstream import SourceLocation


__all__ = [
    "ItemSlotProvider",
    "ClassNameProvider",
    "FunctionSignatureProvider",
    "FunctionSignatureArgProvider",
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


class FunctionSignatureProvider(BaseFeatureProvider[AstFunctionSignature]):
    @classmethod
    def semantics(cls, params):
        signature = params.node
        location = signature.location
        node = AstNode(
            location=location,
            end_location=offset_location(signature.location, len(signature.name)),
        )

        tokens = [(node, "function", [])]

        if len(signature.arguments) >= 1:
            first = signature.arguments[0]

            if isinstance(first, AstFunctionSignatureArgument) and first.name == "self":
                tokens.append((first, "macro", []))

        return tokens


class FunctionSignatureArgProvider(BaseFeatureProvider[AstFunctionSignatureArgument]):
    @classmethod
    def semantics(cls, params):
        if params.node.type_annotation:
            return [(params.node.type_annotation, "class", [])]

        return None


class ValueProvider(BaseFeatureProvider[AstValue]):
    @classmethod
    def semantics(cls, params):
        value = params.node

        metadata = retrieve_metadata(value, VariableMetadata)

        if not metadata or not metadata.type_annotation:
            return None

        annotation = metadata.type_annotation

        if annotation is NoneType:

            return [(value, "variable", ["readonly"])]

        elif annotation is bool:
            return [(value, "macro", [])]

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
