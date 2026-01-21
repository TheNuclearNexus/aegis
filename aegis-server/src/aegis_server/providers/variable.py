from functools import reduce
import inspect
import logging
from typing import Any, get_origin
from aegis_core.ast.features.provider import BaseFeatureProvider, DefinitionParams
from aegis_core.ast.helpers import node_location_to_range, offset_location, _hash_node
from aegis_core.ast.metadata import VariableMetadata, attach_metadata, retrieve_metadata
from aegis_core.reflection import (
    UNKNOWN_TYPE,
    FunctionInfo,
    TypeInfo,
    get_annotation_description,
    get_function_description,
    get_type_info,
    search_scope_for_binding,
)
from aegis_core.semantics import TokenModifier, TokenType
from aegis_core.reflection.type_representation import (
    CallableRepresentation,
    ClassRepresentation,
    InstanceRepresentation,
    ReferencedTypeRepresentation,
)
import lsprotocol.types as lsp
from bolt import (
    AstAttribute,
    AstIdentifier,
    AstImportedItem,
    AstTargetAttribute,
    AstTargetIdentifier,
    Module,
    Runtime,
    Variable,
)
from mecha import AstNode, Mecha


__all__ = ["VariableFeatureProvider"]


def get_type_annotation(resource_location: str, node: AstNode):
    metadata = retrieve_metadata(resource_location, node, VariableMetadata)

    if not metadata:
        return None

    return metadata.type_annotation


def add_variable_definition(
    resource_location: str,
    items: list[lsp.CompletionItem],
    name: str,
    variable: Variable,
):
    possible_types = set()

    for binding in variable.bindings:
        origin = binding.origin
        if annotation := get_type_annotation(resource_location, origin):
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


def get_bolt_completions(resource_location: str, node: AstNode):
    if isinstance(node, AstAttribute):
        node = node.value

    metadata = retrieve_metadata(resource_location, node, VariableMetadata)

    if not metadata:
        return

    type_annotation = metadata.type_annotation

    if type_annotation is UNKNOWN_TYPE:
        return

    # type_info = (
    #     get_type_info(type_annotation)
    #     if not isinstance(type_annotation, TypeInfo)
    #     else type_annotation
    # )

    items = []

    # for name, type in type_info.fields.items():
    #     add_variable_completion(items, name, type)

    # for name, function_info in type_info.functions.items():
    #     items.append(
    #         lsp.CompletionItem(
    #             name,
    #             kind=lsp.CompletionItemKind.Function,
    #             documentation=lsp.MarkupContent(
    #                 kind=lsp.MarkupKind.Markdown,
    #                 value=get_function_description(name, function_info),
    #             ),
    #         )
    #     )

    return items


def generic_variable_token(
    resource_location: str, variable_name: str, identifier: Any
) -> list[tuple[AstNode, TokenType, list[TokenModifier]]]:
    nodes: list[tuple[AstNode, TokenType, list[TokenModifier]]] = []
    annotation = get_type_annotation(resource_location, identifier)

    while isinstance(annotation, ReferencedTypeRepresentation):
        metadata = retrieve_metadata(
            annotation.resource_location, annotation.hash, VariableMetadata
        )
        assert metadata
        annotation = metadata.type_annotation

    if (
        isinstance(annotation, InstanceRepresentation)
        and isinstance(annotation.parent, CallableRepresentation)
        and not isinstance(annotation.parent, ClassRepresentation)
    ):
        annotation = annotation.parent

    if annotation is not None and (isinstance(annotation, ClassRepresentation)):
        nodes.append((identifier, "class", []))

    elif annotation is not None and (isinstance(annotation, CallableRepresentation)):
        nodes.append((identifier, "function", []))
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


def attribute_token(resource_location, node: AstAttribute | AstTargetAttribute):
    temp_node = AstIdentifier(
        offset_location(node.end_location, -len(node.name)),
        node.end_location,
        node.name,
    )

    metadata = retrieve_metadata(resource_location, node, VariableMetadata)
    attach_metadata("temp", temp_node, metadata or VariableMetadata())

    return generic_variable_token(
        "temp",
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

        metadata = retrieve_metadata(params.resource_location, node, VariableMetadata)
        name = (
            node.value
            if not isinstance(node, (AstAttribute, AstTargetAttribute, AstImportedItem))
            else node.name
        )

        if metadata and metadata.type_annotation:

            type_annotation = metadata.type_annotation

            logging.debug(f"\n\n{id(type_annotation)}\n{_hash_node(node)}\n\n")

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
        return get_bolt_completions(params.resource_location, params.node)

    @classmethod
    def semantics(
        cls, params
    ) -> list[tuple[AstNode, TokenType, list[TokenModifier]]] | None:
        match params.node:
            case AstIdentifier():
                return generic_variable_token(
                    params.resource_location, params.node.value, params.node
                )
            case AstTargetIdentifier():
                return generic_variable_token(
                    params.resource_location, params.node.value, params.node
                )
            case AstAttribute():
                return attribute_token(params.resource_location, params.node)
            case AstTargetAttribute():
                return attribute_token(params.resource_location, params.node)
            case AstImportedItem():
                return generic_variable_token(
                    params.resource_location, params.node.name, params.node
                )
        return None

    @classmethod
    def definition(
        cls,
        params: DefinitionParams[
            AstIdentifier
            | AstAttribute
            | AstTargetAttribute
            | AstTargetIdentifier
            | AstImportedItem
        ],
    ) -> list[lsp.Location | lsp.LocationLink] | lsp.Location | lsp.LocationLink | None:
        match params.node:
            case AstIdentifier() as ident:
                var_name = ident.value

                module = params.compilation.module

                if module is None:
                    return

                scope = module.lexical_scope

                result = search_scope_for_binding(var_name, ident, scope)

                if not result:
                    return

                binding, scope = result

                range = node_location_to_range(binding.origin)

                return lsp.Location(params.text_document_uri, range)
