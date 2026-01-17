from functools import reduce
import inspect
from typing import Any, get_origin
from aegis_core.ast.features.provider import BaseFeatureProvider
from aegis_core.ast.helpers import offset_location
from aegis_core.ast.metadata import VariableMetadata, attach_metadata, retrieve_metadata
from aegis_core.reflection import UNKNOWN_TYPE, FunctionInfo, TypeInfo, get_annotation_description, get_function_description, get_type_info
from aegis_core.semantics import TokenModifier, TokenType
import lsprotocol.types as lsp
from bolt import (
    AstAttribute,
    AstIdentifier,
    AstImportedItem,
    AstTargetAttribute,
    AstTargetIdentifier,
    Variable,
)
from mecha import AstNode



__all__ = ["VariableFeatureProvider"]


def get_type_annotation(node: AstNode):
    metadata = retrieve_metadata(node, VariableMetadata)

    if not metadata:
        return None

    return metadata.type_annotation


def add_variable_definition(
    items: list[lsp.CompletionItem], name: str, variable: Variable
):
    possible_types = set()

    for binding in variable.bindings:
        origin = binding.origin
        if annotation := get_type_annotation(origin):
            possible_types.add(annotation)

    if len(possible_types) > 0:
        _type = reduce(lambda a, b: a | b, possible_types)
        add_variable_completion(items, name, _type)


def add_raw_definition(items: list[lsp.CompletionItem], name: str, value: Any):
    if inspect.isclass(value) or isinstance(value, TypeInfo):
        add_class_completion(items, name, value)
    elif (
        inspect.isfunction(value)
        or inspect.isbuiltin(value)
        or isinstance(value, FunctionInfo)
    ):
        add_function_completion(items, name, value)
    else:
        add_variable_completion(items, name, type(value))


def add_class_completion(
    items: list[lsp.CompletionItem], name: str, type_annotation: Any
):
    description = get_annotation_description(name, type_annotation)
    documentation = lsp.MarkupContent(lsp.MarkupKind.Markdown, description)

    items.append(
        lsp.CompletionItem(
            name, documentation=documentation, kind=lsp.CompletionItemKind.Class
        )
    )


def add_function_completion(items: list[lsp.CompletionItem], name: str, function: Any):
    description = get_annotation_description(name, function)
    documentation = lsp.MarkupContent(lsp.MarkupKind.Markdown, description)

    items.append(
        lsp.CompletionItem(
            name, documentation=documentation, kind=lsp.CompletionItemKind.Function
        )
    )


def add_variable_completion(
    items: list[lsp.CompletionItem], name: str, type_annotation: Any
):
    kind = (
        lsp.CompletionItemKind.Property
        if not name.isupper()
        else lsp.CompletionItemKind.Constant
    )

    description = get_annotation_description(name, type_annotation)
    documentation = lsp.MarkupContent(lsp.MarkupKind.Markdown, description)

    items.append(lsp.CompletionItem(name, documentation=documentation, kind=kind))


def get_bolt_completions(node: AstNode):
    if isinstance(node, AstAttribute):
        node = node.value

    metadata = retrieve_metadata(node, VariableMetadata)

    if not metadata:
        return

    type_annotation = metadata.type_annotation

    if type_annotation is UNKNOWN_TYPE:
        return

    type_info = (
        get_type_info(type_annotation)
        if not isinstance(type_annotation, TypeInfo)
        else type_annotation
    )

    items = []

    for name, type in type_info.fields.items():
        add_variable_completion(items, name, type)

    for name, function_info in type_info.functions.items():
        items.append(
            lsp.CompletionItem(
                name,
                kind=lsp.CompletionItemKind.Function,
                documentation=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value=get_function_description(name, function_info),
                ),
            )
        )

    return items


def generic_variable_token(
    variable_name: str, identifier: Any
) -> list[tuple[AstNode, TokenType, list[TokenModifier]]]:
    nodes: list[tuple[AstNode, TokenType, list[TokenModifier]]] = []
    annotation = get_type_annotation(identifier)

    if annotation is not None and (
        inspect.isfunction(annotation)
        or inspect.isbuiltin(annotation)
        or isinstance(annotation, FunctionInfo)
    ):
        nodes.append((identifier, "function", []))
    elif annotation is not None and (
        get_origin(annotation) is type or isinstance(annotation, TypeInfo)
    ):
        nodes.append((identifier, "class", []))
    else:
        kind = "variable"
        modifiers: list[TokenModifier] = []

        if variable_name.isupper():
            modifiers.append("readonly")
        elif variable_name == "self":
            kind = "macro"

        nodes.append(
            (
                identifier,
                kind,
                modifiers,
            )
        )

    return nodes


def attribute_token(node: AstAttribute | AstTargetAttribute):
    temp_node = AstIdentifier(
        offset_location(node.end_location, -len(node.name)),
        node.end_location,
        node.name,
    )

    metadata = retrieve_metadata(temp_node, VariableMetadata)
    attach_metadata(temp_node, metadata or VariableMetadata())

    return generic_variable_token(
        node.name,
        temp_node,
    )


class VariableFeatureProvider(
    BaseFeatureProvider[
        AstIdentifier
        | AstAttribute
        | AstTargetAttribute
        | AstTargetIdentifier
        | AstImportedItem
    ]
):
    @classmethod
    def hover(cls, params) -> lsp.Hover | None:
        node = params.node
        text_range = params.text_range

        metadata = retrieve_metadata(node, VariableMetadata)
        name = (
            node.value
            if not isinstance(node, (AstAttribute, AstTargetAttribute, AstImportedItem))
            else node.name
        )

        if metadata and metadata.type_annotation:

            type_annotation = metadata.type_annotation

            description = get_annotation_description(name, type_annotation)

            return lsp.Hover(
                lsp.MarkupContent(lsp.MarkupKind.Markdown, description), text_range
            )

        return lsp.Hover(
            lsp.MarkupContent(
                lsp.MarkupKind.Markdown,
                f"```python\n(variable) {name}\n```",
            ),
            text_range,
        )

    @classmethod
    def completion(cls, params):
        return get_bolt_completions(params.node)

    @classmethod
    def semantics(
        cls, params
    ) -> list[tuple[AstNode, TokenType, list[TokenModifier]]] | None:
        match params.node:
            case AstIdentifier():
                return generic_variable_token(params.node.value, params.node)
            case AstTargetIdentifier():
                return generic_variable_token(params.node.value, params.node)
            case AstAttribute():
                return attribute_token(params.node)
            case AstTargetAttribute():
                return attribute_token(params.node)
            case AstImportedItem():
                return generic_variable_token(params.node.name, params.node)
        return None