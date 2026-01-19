from typing import Protocol
from beet.core.utils import extra_field
from dataclasses import dataclass, field


class TypeRepresentation: 
    def description(self, variable_name: str) -> str:
        return f"```python\n(variable) {variable_name}\n````"


class DocumentedTypeRepresentation(Protocol):
    doc_string: str | None


class NoneRepresentation(TypeRepresentation): ...


@dataclass
class PrimitiveRepresentation(TypeRepresentation):
    primitive: (
        type[int]
        | type[float]
        | type[str]
        | type[bool]
        | type[complex]
        | type[list]
        | type[dict]
        | type[set]
    ) = field()

@dataclass
class FunctionRepresentation(TypeRepresentation):
    arguments: list[tuple[str, TypeRepresentation]] = field()
    return_type: TypeRepresentation = field(default_factory=NoneRepresentation)
    doc_string: str | None = extra_field()


@dataclass
class ClassRepresentation(TypeRepresentation):
    name: str = field()
    fields: list[tuple[str, TypeRepresentation]] = field()
    methods: list[tuple[str, TypeRepresentation]] = field()
    doc_string: str | None = extra_field()

    sub_classes: list[ClassRepresentation] = field(default_factory=list)
