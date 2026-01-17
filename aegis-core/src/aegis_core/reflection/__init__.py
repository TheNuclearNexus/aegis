import inspect
import types
import typing
from copy import copy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_args, get_origin

UNKNOWN_TYPE = object()
TYPE_TO_INFO: dict[type, "TypeInfo"] = {}


@dataclass
class ParameterInfo:
    annotation: Any
    default: Any


@dataclass
class FunctionInfo:
    parameters: list[tuple[str, ParameterInfo]]
    return_annotation: Any
    doc: str | None

    @staticmethod
    def from_signature(
        signature: inspect.Signature, doc: str | None, hints: dict[str, Any]
    ) -> "FunctionInfo":
        return FunctionInfo(
            parameters=[
                (
                    name,
                    ParameterInfo(
                        annotation=parameter.annotation, default=parameter.default
                    ),
                )
                for name, parameter in signature.parameters.items()
            ],
            return_annotation=hints.get("return") or signature.return_annotation,
            doc=doc,
        )

    @staticmethod
    def extract(field_value: Any):
        try:
            hints = typing.get_type_hints(field_value)
            signature = inspect.signature(field_value)
            return FunctionInfo.from_signature(signature, field_value.__doc__, hints)
        except Exception as e:
            return FunctionInfo(
                parameters=[
                    (
                        "???",
                        ParameterInfo(inspect.Parameter.empty, inspect.Parameter.empty),
                    )
                ],
                return_annotation=UNKNOWN_TYPE,
                doc="Error while extracting signature: " + str(e),
            )

    def __hash__(self) -> int:
        return hash(self.__repr__())


@dataclass
class TypeInfo:
    doc: str | None
    fields: dict[str, Any] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)

    def get_member(self, field_name: str) -> Any | FunctionInfo:
        return self.fields.get(field_name) or self.functions.get(field_name)

    def add_member(
        self,
        field_annotations: dict[str, Any],
        field_name: str,
        field_value: Any,
        skip_fields=False,
    ):
        if (
            inspect.isfunction(field_value)
            or inspect.ismethod(field_value)
            or inspect.ismethoddescriptor(field_value)
            or inspect.isbuiltin(field_value)
        ):
            self.functions[field_name] = FunctionInfo.extract(field_value)

        elif skip_fields:
            return
        elif field_name in field_annotations:
            self.fields[field_name] = field_annotations[field_name]
        elif _ := get_origin(field_value):
            self.fields[field_name] = field_value
        else:
            self.fields[field_name] = type(field_value)

    def __hash__(self) -> int:
        return hash(self.__repr__())


def get_type_info(_type: type) -> TypeInfo:

    if _type in TYPE_TO_INFO:
        return TYPE_TO_INFO[type]

    info = TypeInfo(_type.__doc__)

    # logging.debug("\n\n")
    # logging.debug("-" * 50)
    # logging.debug("Indexing type " + _type.__name__)

    field_annotations = (
        inspect.get_annotations(_type)
        if inspect.isclass(_type) or callable(_type) or inspect.ismodule(_type)
        else {}
    )

    if is_dataclass(_type):
        handled_fields = set()
        for field in fields(_type):
            info.add_member({}, field.name, field.type)
            handled_fields.add(field.name)
        for field_name, field_value in inspect.getmembers(_type):
            if field_name in handled_fields:
                continue

            # field_value = getattr(_type, field_name)
            info.add_member(field_annotations, field_name, field_value)
    else:
        for field_name, field_value in inspect.getmembers(_type):
            # field_value = getattr(_type, field_name)
            # logging.debug(f"{field_name}, {field_value}")
            info.add_member(field_annotations, field_name, field_value)

    # logging.debug("-" * 50)
    # logging.debug("\n\n")
    # logging.debug(info)
    return info


def get_name_of_type(annotation):
    if annotation is inspect.Parameter.empty:
        return None

    if annotation is types.NoneType:
        return "None"

    if annotation is UNKNOWN_TYPE:
        return "???"

    origin = typing.get_origin(annotation)
    args = list(map(str, map(get_name_of_type, typing.get_args(annotation))))

    if origin is typing.Union or origin is types.UnionType:
        return " | ".join(args)
    elif origin:
        return f"{origin.__name__}[{', '.join(args)}]"

    try:
        return annotation.__repr__()
    except:
        return annotation.__name__ if hasattr(annotation, "__name__") else annotation


def format_function_hints(
    name: str,
    signature: FunctionInfo,
    keyword: str = "def",
    show_return_type: bool = True,
):
    hint = f"{keyword} {name}("

    return_type = signature.return_annotation

    parameters = []

    for name, parameter in signature.parameters:
        annotation = get_name_of_type(parameter.annotation)

        if annotation is None and parameter.default is not inspect.Parameter.empty:
            annotation = get_name_of_type(type(parameter.default))

        annotation_string = ": " + str(annotation) if annotation else ""
        default_string = (
            " = " + parameter.default.__repr__()  # type: ignore
            if parameter.default is not inspect.Parameter.empty
            else ""
        )

        parameters.append(f"\n\t{name}{annotation_string}{default_string}")

    hint += ",".join(parameters)

    hint += f"\n)"

    if show_return_type and signature.return_annotation is not types.NoneType:
        hint += f" -> {get_name_of_type(return_type)}"

    return hint


def get_doc_string(doc: Any):
    return "\n---\n" + doc if isinstance(doc, str) else ""


def get_variable_description(name: str, value: Any):
    if inspect.isclass(value):
        return f"```python\n(variable) {name}: {get_name_of_type(value)}\n```"

    doc_string = get_doc_string(value.__doc__)
    return f"```python\n(variable) {name}: {get_name_of_type(type(value))}\n```{doc_string}"


def get_class_description(name: str, value: type | TypeInfo):
    if not isinstance(value, TypeInfo):
        value = get_type_info(value)

    doc_string = get_doc_string(value.doc)

    if not (init := value.functions.get("__init__")):
        return f"```python\nclass {name}()\n```{doc_string}"

    init = copy(init)

    if len(init.parameters) > 0:
        init.parameters.pop(0)

    return f"```python\n{format_function_hints(name, init, keyword='class', show_return_type=False)}\n```{doc_string}"


def get_function_description(name: str, function: Any):
    function_info = None
    if isinstance(function, FunctionInfo):
        function_info = function
    else:
        function_info = FunctionInfo.extract(function)

    doc_string = get_doc_string(function_info.doc)

    return f"```py\n{format_function_hints(name, function_info)}\n```{doc_string}"


def get_annotation_description(name: str, type_annotation: Any):
    if get_origin(type_annotation) is type:
        args = get_args(type_annotation)
        description = get_class_description(name, args[0])
    elif isinstance(type_annotation, TypeInfo):
        description = get_class_description(name, type_annotation)
    elif (
        inspect.isfunction(type_annotation)
        or inspect.isbuiltin(type_annotation)
        or isinstance(type_annotation, FunctionInfo)
    ):
        description = get_function_description(name, type_annotation)
    else:
        description = get_variable_description(name, type_annotation)

    return description
