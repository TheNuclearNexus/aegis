import builtins
import inspect
import logging
from typing import Any, Protocol, runtime_checkable
import typing
from ..ast.helpers import BaseMetadata, retrieve_metadata, _hash_node
from beet.core.utils import extra_field
from dataclasses import dataclass, field

from bolt import (
    AstExpression,
    AstFunctionSignature,
    AstFunctionSignatureArgument,
    AstIdentifier,
    AstTypeAnnotation,
    AstValue,
)
from mecha import AstNode, Serializer


_reprs: dict[Any, TypeRepresentation] = dict()


@dataclass
class TypeRepresentation:
    def description(self, variable_name: str) -> str:
        return f"```python\n(variable) {variable_name}\n````"

    def pretty(self) -> str:
        return "???"

    @staticmethod
    def from_python(variable: Any) -> TypeRepresentation:
        if id(variable) in _reprs:
            return _reprs[id(variable)]

        repr = UNKNOWN_TYPE

        _reprs[id(variable)] = UNKNOWN_TYPE

        if inspect.isbuiltin(variable):
            repr = BuiltinRepresentation(variable)
        elif inspect.isfunction(variable):
            repr = FunctionRepresentation.from_python(variable)
        elif inspect.ismethod(variable):
            repr = FunctionRepresentation.from_python(variable)
        elif inspect.ismethoddescriptor(variable):
            repr = FunctionRepresentation.from_python(variable)
        elif inspect.ismethodwrapper(variable):
            repr = CallableRepresentation.from_python(variable)
        elif inspect.isclass(variable):
            repr = ClassRepresentation.from_python(variable)
        elif inspect.ismodule(variable):
            repr = ModuleRepresentation.from_python(variable)

        _reprs[id(variable)] = repr

        return repr


UNKNOWN_TYPE = TypeRepresentation()


@runtime_checkable
class DocumentedTypeRepresentation(Protocol):
    doc_string: str | None


class NoneRepresentation(TypeRepresentation): ...


@dataclass
class BuiltinRepresentation(TypeRepresentation):
    primitive: Any = field()


@dataclass
class CallableRepresentation(TypeRepresentation):
    name: str = field()
    doc_string: str | None = field()

    @staticmethod
    def from_python(variable: Any) -> TypeRepresentation:
        return CallableRepresentation(
            name=variable.__name__,
            doc_string=variable.__doc__ if hasattr(variable, "__doc__") else None,
        )

    def description(self, variable_name: str) -> str:
        return (
            f"```py\n(callable) {variable_name}\n```\n" + f"---\n{self.doc_string}"
            if self.doc_string
            else ""
        )


@dataclass
class ParameterRepresentation(TypeRepresentation):
    name: str
    annotation: TypeRepresentation
    default: Any

    def pretty(self) -> str:
        return (
            f"{self.name}"
            + (
                f": {self.annotation.pretty()}"
                if self.annotation is not UNKNOWN_TYPE
                else ""
            )
            + (f" = {self.default}" if self.default else "")
        )


@dataclass
class InstanceRepresentation(TypeRepresentation):
    parent: TypeRepresentation = field()

    def description(self, variable_name: builtins.str) -> builtins.str:
        return f"```py\n(variable) {variable_name}: {self.parent.pretty()}\n```" + (
            "\n---\n" + self.parent.doc_string
            if isinstance(self.parent, DocumentedTypeRepresentation)
            and self.parent.doc_string
            else ""
        )


@dataclass
class FunctionRepresentation(CallableRepresentation, InstanceRepresentation):
    parent: TypeRepresentation = field(
        init=False, default_factory=lambda: UNKNOWN_TYPE
    )  # type:ignore

    arguments: list[ParameterRepresentation] = field()
    var_arguments: list[ParameterRepresentation] = field()
    var_kw_arguments: list[ParameterRepresentation] = field()
    return_type: TypeRepresentation = field(default_factory=NoneRepresentation)

    @staticmethod
    def from_python(variable: Any):
        try:
            signature = inspect.signature(variable)
        except:
            return CallableRepresentation.from_python(variable)

        arguments = []
        var_arguments = []
        var_kw_arguments = []

        return_type = UNKNOWN_TYPE

        for name, param in signature.parameters.items():
            annotation = TypeRepresentation.from_python(param.annotation)

            match param.kind:
                case inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    arguments.append(
                        ParameterRepresentation(name, annotation, param.default)
                    )
                case inspect.Parameter.POSITIONAL_ONLY:
                    arguments.append(
                        ParameterRepresentation(name, annotation, param.default)
                    )
                case inspect.Parameter.VAR_POSITIONAL:
                    var_arguments.append(
                        ParameterRepresentation(name, annotation, param.default)
                    )
                case inspect.Parameter.VAR_KEYWORD:
                    var_kw_arguments.append(
                        ParameterRepresentation(name, annotation, param.default)
                    )

        if signature.return_annotation:
            return_type = TypeRepresentation.from_python(signature.return_annotation)

        return FunctionRepresentation(
            name=variable.__name__,
            arguments=arguments,
            var_arguments=var_arguments,
            var_kw_arguments=var_kw_arguments,
            return_type=return_type,
            doc_string=variable.__doc__,
        )

    def description(self, variable_name: str) -> str:
        args = ", ".join(
            [a.pretty() for a in self.arguments]
            + ["*" + a.pretty() for a in self.var_arguments]
            + ["**" + a.pretty() for a in self.var_kw_arguments]
        )

        return_type = (
            " -> " + self.return_type.pretty()
            if self.return_type is not UNKNOWN_TYPE
            else ""
        )

        return f"```py\ndef {self.name}({args}){return_type}\n```\n" + (
            f"---\n{self.doc_string}" if self.doc_string else ""
        )


@dataclass
class ClassRepresentation(CallableRepresentation):
    fields: list[tuple[str, TypeRepresentation, Any]] = field()
    methods: list[tuple[str, TypeRepresentation]] = field()

    generics: list[TypeRepresentation] = field()
    bases: list[ClassRepresentation] = field(default_factory=list)

    @staticmethod
    def from_python(variable: Any):
        fields = []
        methods = []

        for member in inspect.getmembers(variable):
            kind = TypeRepresentation.from_python(member[1])

            if isinstance(kind, FunctionRepresentation):
                methods.append((member[0], kind))
            else:
                fields.append((member[0], kind, member[1]))

        doc_string = inspect.getdoc(variable)

        return ClassRepresentation(
            name=variable.__name__,
            fields=fields,
            methods=methods,
            generics=[],
            doc_string=doc_string,
        )

    def description(self, variable_name: str) -> str:
        return (
            f"```py\n(class) {self.pretty()}\n```\n" + f"---\n{self.doc_string}"
            if self.doc_string
            else ""
        )

    def pretty(self) -> str:
        generics = [t.pretty() for t in self.generics]
        return f"{self.name}{f'[{', '.join(generics)}]' if len(generics) > 0 else ''}"

    def get_field(self, variable: str) -> tuple[str, TypeRepresentation, Any] | None:
        for field in self.fields:
            if field[0] == variable:
                return field

        for base in self.bases:
            if base == self:
                continue

            if field := base.get_field(variable):
                return field

    def get_method(self, variable: str) -> tuple[str, TypeRepresentation] | None:
        for method in self.methods:
            if method[0] == variable:
                return method

        for base in self.bases:
            if base == self:
                continue

            if method := base.get_method(variable):
                return method

        return None

@dataclass
class ModuleRepresentation(TypeRepresentation):
    doc_string: str|None = field()
    members: list[tuple[str, TypeRepresentation]] = field()

    @staticmethod
    def from_python(variable: Any) -> TypeRepresentation:
        members = []
        for name in dir(variable):
            attr = getattr(variable, name)
            members.append((name, TypeRepresentation.from_python(attr)))
        
        return ModuleRepresentation(variable.__doc__, members)

    def get_member(self, name: str) -> TypeRepresentation | None:
        for member in self.members:
            if member[0] == name:
                return member[1]

        return None


@dataclass
class UnionRepresentation(TypeRepresentation):
    types: list[TypeRepresentation]


@dataclass
class ReferencedTypeRepresentation(TypeRepresentation):
    resource_location: str
    hash: int

    def get_reference(self) -> TypeRepresentation:
        metadata = retrieve_metadata(self.resource_location, self.hash, BaseMetadata)
        assert metadata is not None
        if metadata.type_annotation != self: #type: ignore
            return metadata.type_annotation  #type: ignore
        else:
            return UNKNOWN_TYPE

    def description(self, variable_name: str) -> str:
        return self.get_reference().description(variable_name)

    def pretty(self) -> str:
        return self.get_reference().pretty()

    @staticmethod
    def from_node(resource_location: str, node: AstNode):
        node_hash = _hash_node(node)

        return ReferencedTypeRepresentation(resource_location, node_hash)
