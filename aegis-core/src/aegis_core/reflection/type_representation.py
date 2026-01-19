import builtins
import inspect
import logging
from typing import Any, Protocol
import typing
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
            repr = CallableRepresentation.from_python(variable)
        elif inspect.ismethodwrapper(variable):
            repr = CallableRepresentation.from_python(variable)
        elif inspect.isclass(variable):
            repr = ClassRepresentation.from_python(variable)

        _reprs[id(variable)] = repr

        return repr


UNKNOWN_TYPE = TypeRepresentation()


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
            + (f": {self.annotation.pretty()}" if self.annotation is not UNKNOWN_TYPE else "")
            + (f" = {self.default}" if self.default else "")
        )


@dataclass
class FunctionRepresentation(CallableRepresentation):
    arguments: list[ParameterRepresentation] = field()
    var_arguments: list[ParameterRepresentation] = field()
    var_kw_arguments: list[ParameterRepresentation] = field()
    return_type: TypeRepresentation = field(default_factory=NoneRepresentation)

    @staticmethod
    def from_python(variable: Any):
        signature = inspect.signature(variable)

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
    fields: list[tuple[str, TypeRepresentation]] = field()
    methods: list[tuple[str, TypeRepresentation]] = field()

    generics: list[TypeRepresentation] = field()
    sub_classes: list[ClassRepresentation] = field(default_factory=list)

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


@dataclass
class InstanceRepresentation(TypeRepresentation):
    parent: TypeRepresentation = field()

    def description(self, variable_name: builtins.str) -> builtins.str:
        return f"```py\n(variable) {variable_name}: {self.parent.pretty()}\n```"


@dataclass
class UnionRepresentation(TypeRepresentation):
    types: list[TypeRepresentation]
