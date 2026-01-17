from dataclasses import replace
from functools import partial
import logging
from typing import Callable, TypeVar, cast

from mecha import AstNode, AstResourceLocation, Rule
from mecha.contrib.nested_location import AstNestedLocation, NestedLocationTransformer
from mecha.contrib.relative_location import (
    RelativeResourceLocationParser,
    resolve_using_database,
)
from tokenstream import TokenStream

from aegis_core.ast.metadata import (
    ResourceLocationMetadata,
    attach_metadata,
    retrieve_metadata,
)

T = TypeVar("T", bound=AstNode)


def set_dict(a: T, b: AstNode) -> T:
    if metadata := retrieve_metadata(b):
        attach_metadata(a, metadata)

    return a


def parse_relative_path(
    self: RelativeResourceLocationParser, stream: TokenStream
) -> AstResourceLocation:
    node: AstResourceLocation = self.parser(stream)
    metadata = (
        retrieve_metadata(node, ResourceLocationMetadata) or ResourceLocationMetadata()
    )


    if node.namespace is None and node.path.startswith(("./", "../")):
        unresolved = node.path
        namespace, resolved = resolve_using_database(
            relative_path=node.path,
            database=self.database,
            location=node.location,
            end_location=node.end_location,
        )

        node = replace(node, namespace=namespace, path=resolved)

        metadata.unresolved_path = unresolved
    else:
        metadata.unresolved_path = node.get_canonical_value()

    attach_metadata(node, metadata)

    return node


def wrap_nested_location(
    original_rule: Callable[
        [NestedLocationTransformer, AstNestedLocation], AstResourceLocation
    ],
    self: NestedLocationTransformer,
    node: AstNestedLocation,
):
    new_node = original_rule(self, node)
    return set_dict(new_node, node)


def apply_patches():
    RelativeResourceLocationParser.__call__ = parse_relative_path
    nested_location_rule = cast(Rule, NestedLocationTransformer.nested_location)
    nested_location_rule.callback = partial(
        wrap_nested_location, nested_location_rule.callback
    )
