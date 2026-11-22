from textwrap import dedent

import pytest
from beet import run_beet
from bolt import AstIdentifier, AstTargetIdentifier, Module, Runtime
from mecha import CompilationUnit, Mecha

from aegis_core.ast.features import HoverParams
from aegis_core.ast.helpers import node_location_to_range
from aegis_core.ast.metadata import clear_metadata
from aegis_core.indexing.project_index import AegisProjectIndex
from aegis_server.providers.variable import VariableFeatureProvider
from aegis_server.server.indexing import TypeAnnotationResolver


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            '''\
            ON_START_GAME = ~/on_start_game
            """Game has started."""
            ON_END_GAME = ~/on_end_game
            """Game has ended.
            A winning team has been chosen.
            """
            start = ON_START_GAME
            end = ON_END_GAME
            ON_END_GAME = ~/other
            other = ON_END_GAME
            ''',
            {
                ("ON_START_GAME", 1): "Game has started.",
                ("ON_END_GAME", 3): (
                    "Game has ended.\nA winning team has been chosen.\n"
                ),
                ("ON_START_GAME", 7): "Game has started.",
                ("ON_END_GAME", 8): (
                    "Game has ended.\nA winning team has been chosen.\n"
                ),
                ("ON_END_GAME", 9): None,
                ("ON_END_GAME", 10): None,
            },
        ),
        (
            '''\
            count: int = 1
            """Number of players."""
            value = count
            ''',
            {
                ("count", 1): "Number of players.",
                ("count", 3): "Number of players.",
            },
        ),
        (
            '''\
            count = 1
            """Global player count."""
            def handler():
                """Handle the event."""
                count = 2
                """Local player count."""
                value = count
            value = count
            ''',
            {
                ("count", 1): "Global player count.",
                ("count", 5): "Local player count.",
                ("count", 7): "Local player count.",
                ("count", 8): "Global player count.",
            },
        ),
    ],
    ids=["resource-locations", "annotated", "nested-scope"],
)
def test_variable_docstring_hover(tmp_path, source, expected):
    resource_location = "test:variable_docstrings"
    clear_metadata(resource_location)

    try:
        with run_beet(directory=tmp_path, config={"require": ["bolt"]}) as ctx:
            mecha = ctx.inject(Mecha)
            runtime = ctx.inject(Runtime)
            file = Module(dedent(source))
            source_path = tmp_path / "variables.bolt"
            index = ctx.inject(AegisProjectIndex)
            index[Module].add_definition(resource_location, source_path)
            mecha.database[file] = CompilationUnit(resource_location=resource_location)
            mecha.database.current = file
            module = runtime.modules[file]

            TypeAnnotationResolver(ctx=ctx, index=index, runtime=runtime, mecha=mecha)(
                module.ast
            )

            hovers = {}
            for node in module.ast.walk():
                if not isinstance(node, (AstIdentifier, AstTargetIdentifier)):
                    continue
                key = (node.value, node.location.lineno)
                if key not in expected:
                    continue
                hover = VariableFeatureProvider.hover(
                    HoverParams(
                        ctx, node, resource_location, node_location_to_range(node)
                    )
                )
                hovers[key] = hover.contents.value

            assert hovers.keys() == expected.keys()
            for key, docstring in expected.items():
                if docstring is None:
                    assert "---" not in hovers[key]
                else:
                    assert hovers[key].endswith(f"\n---\n{docstring}")
    finally:
        clear_metadata(resource_location)
