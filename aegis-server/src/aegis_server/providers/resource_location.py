__all__ = ["ResourceLocationFeatureProvider"]

import logging
import os
from pathlib import Path
import re
from typing import cast

from aegis_core.ast.features.provider import BaseFeatureProvider
from aegis_core.ast.helpers import node_location_to_range
from aegis_core.ast.metadata import ResourceLocationMetadata, retrieve_metadata
from aegis_core.indexing.project_index import AegisProjectIndex
import lsprotocol.types as lsp
from beet import File, NamespaceFile
from mecha import AstResourceLocation

from aegis_core.registry import AegisGameRegistries

def add_registry_items(
    registries: AegisGameRegistries,
    represents: str,
    prefix: str = "",
    kind: lsp.CompletionItemKind = lsp.CompletionItemKind.Value,
):
    if represents in registries:
        registry_items = registries[represents]

        return [
            lsp.CompletionItem(prefix + "minecraft:" + k, kind=kind, sort_text=k)
            for k in registry_items
        ]

    return []


def get_path(path: str) -> tuple[str | None, Path]:
    segments = path.split(":")
    if len(segments) == 1:
        return (None, Path(segments[0]))
    else:
        return (segments[0], Path(segments[1]))


class ResourceLocationFeatureProvider(BaseFeatureProvider[AstResourceLocation]):
    @classmethod
    def hover(cls, params) -> lsp.Hover | None:
        node = params.node
        text_range = params.text_range

        metadata = retrieve_metadata(node, ResourceLocationMetadata)

        if metadata is None or metadata.represents is None:
            path_type = None
        elif isinstance(metadata.represents, str):
            path_type = metadata.represents
        elif issubclass(metadata.represents, File):
            path_type = metadata.represents.snake_name

        type_line = f"**{path_type}**\n" if path_type else ""

        return lsp.Hover(
            lsp.MarkupContent(
                lsp.MarkupKind.Markdown,
                f"{type_line}```yaml\n{node.get_canonical_value()}\n```",
            ),
            text_range,
        )

    @classmethod
    def definition(cls, params):
        project_index = params.ctx.inject(AegisProjectIndex)
        node = params.node

        metadata = retrieve_metadata(node, ResourceLocationMetadata)

        if not metadata or not metadata.represents:
            return

        if isinstance(metadata.represents, str):
            return

        path = node.get_canonical_value()
        definitions = project_index[metadata.represents].get_definitions(path)

        return [
            lsp.LocationLink(
                target_uri=Path(path).as_uri(),
                target_range=node_location_to_range(location),
                target_selection_range=node_location_to_range(location),
                origin_selection_range=node_location_to_range(node),
            )
            for path, *location in definitions
        ]

    @classmethod
    def references(cls, params) -> list[lsp.Location] | None:
        project_index = params.ctx.inject(AegisProjectIndex)
        node = params.node

        metadata = retrieve_metadata(node, ResourceLocationMetadata)

        if not metadata or not metadata.represents:
            return

        if isinstance(metadata.represents, str):
            return

        path = node.get_canonical_value()
        references = project_index[metadata.represents].get_references(path)

        return [
            lsp.Location(Path(path).as_uri(), node_location_to_range(location))
            for path, *location in references
        ]

    @classmethod
    def completion(cls, params):
        node = params.node
        project_index = params.ctx.inject(AegisProjectIndex)

        metadata = retrieve_metadata(node, ResourceLocationMetadata)

        if not metadata or not metadata.represents:
            return

        represents = metadata.represents

        if not isinstance(represents, str):

            path = node.get_canonical_value()

            if node.is_tag:
                path = path[1:]

            resolved = get_path(path)

            if not metadata.unresolved_path:
                return

            unresolved = get_path(metadata.unresolved_path)

            if unresolved[1].name == "~" or metadata.unresolved_path.endswith("/"):
                resolved_parent = resolved[1]
                unresolved_parent = unresolved[1]
            else:
                resolved_parent = resolved[1].parent
                unresolved_parent = unresolved[1].parent

            items = []

            for file in project_index[represents]:
                file_path = get_path(file)

                if not (
                    file_path[0] == resolved[0]
                ):
                    continue
                
                if unresolved[0]:
                    if not file_path[1].is_relative_to(resolved_parent):
                        continue

                    relative = file_path[1].relative_to(resolved_parent)
                else:
                    relative = os.path.relpath(file_path[1], resolved_parent)

                if unresolved[0] is None and unresolved[1].name == "":
                    new_path = "./" + str(relative)
                else:
                    new_path = str(unresolved_parent / relative)

                insert_text = (
                    f"{unresolved[0] + ':' if unresolved[0] else ''}{new_path}"
                ).replace("\\", "/")
                
                if node.is_tag:
                    insert_text = "#" + insert_text

                height_above = len(re.compile(r"\.\./").findall(str(relative)))

                items.append(
                    lsp.CompletionItem(
                        label=insert_text,
                        documentation=file,
                        sort_text=str(height_above),
                        text_edit=lsp.InsertReplaceEdit(
                            insert_text,
                            node_location_to_range(node),
                            node_location_to_range(node),
                        ),
                    )
                )

            return lsp.CompletionList(False, items)

        else:
            registries = params.ctx.inject(AegisGameRegistries)
            items = []

            items.extend(add_registry_items(registries, represents))
            items.extend(
                add_registry_items(
                    registries,
                    "tag/" + represents,
                    "#",
                    lsp.CompletionItemKind.Constant,
                )
            )

            return lsp.CompletionList(False, items)
        
    @classmethod
    def semantics(cls, params):
        return [(params.node, "function", [])]

