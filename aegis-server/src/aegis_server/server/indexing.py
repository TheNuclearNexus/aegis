import builtins
import inspect
import logging
import traceback
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from types import ModuleType
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from beet import (
    Advancement,
    Function,
    LootTable,
    NamespaceFile,
    Predicate,
)
from beet.core.utils import extra_field, required_field
from bolt import (
    AstAssignment,
    AstAttribute,
    AstCall,
    AstClassName,
    AstDict,
    AstDocstring,
    AstExpressionBinary,
    AstFromImport,
    AstFunctionSignature,
    AstFunctionSignatureArgument,
    AstFunctionSignatureVariadicArgument,
    AstFunctionSignatureVariadicKeywordArgument,
    AstIdentifier,
    AstImportedItem,
    AstList,
    AstStatement,
    AstTargetIdentifier,
    AstTuple,
    AstTypeDeclaration,
    AstValue,
    Binding,
    CompiledModule,
    LexicalScope,
    Module,
    Runtime,
)
from mecha import (
    AbstractNode,
    AstBlock,
    AstChildren,
    AstCommand,
    AstError,
    AstItemStack,
    AstJson,
    AstNode,
    AstParticle,
    AstResourceLocation,
    AstRoot,
    AstSelectorArgument,
    CompilationError,
    Mecha,
    MutatingReducer,
    Reducer,
    rule,
)
from mecha.contrib.nested_location import (
    AstNestedLocation,
    NestedLocationResolver,
    NestedLocationTransformer,
)

from aegis_core.ast.metadata import (
    ResourceLocationMetadata,
    VariableMetadata,
    attach_metadata,
    retrieve_metadata,
)
from aegis_core.indexing.project_index import AegisProjectIndex, valid_resource_location
from aegis_core.reflection import (
    UNKNOWN_TYPE,
    FunctionInfo,
    ParameterInfo,
    TypeInfo,
    get_type_info,
)

from .shadows.compile_document import COMPILATION_RESULTS
from .shadows.context import LanguageServerContext

Node = TypeVar("Node", bound=AstNode)


def node_to_types(node: AstNode):

    types = []
    for n in node.walk():
        if isinstance(n, AstExpressionBinary) and n.operator == "|":
            continue

        annotation = expression_to_annotation(n)

        if annotation is not UNKNOWN_TYPE:
            types.append(annotation)

    if len(types) == 0:
        return UNKNOWN_TYPE
    elif len(types) == 1:
        return types[0]

    # TODO: Fix type unions
    return []
    return reduce(lambda a, b: a | b, types)


def expression_to_annotation(expression):
    type_annotation = UNKNOWN_TYPE
    match (expression):
        case AstValue() as value:
            type_annotation = type(value.value)
        case AstDict() as _dict:
            type_annotation = dict
        case AstList():
            type_annotation = list
        case AstTuple():
            type_annotation = tuple
        case AstIdentifier():
            identifier_type = get_type_annotation(expression)
            if get_origin(identifier_type) is not type:
                return UNKNOWN_TYPE

            type_annotation = get_args(identifier_type)[0]
    return type_annotation


def are_equal(a: AstNode, b: AstNode) -> bool:
    # logging.debug(f"{a} == {b}")
    # logging.debug(f"{a.location} == {b.location}")
    # logging.debug(f"{a.end_location} == {b.end_location}")

    return a.location == b.location and a.end_location == b.end_location


def was_referenced(references: list[AstNode], identifier: AstNode):
    for reference in references:
        if are_equal(identifier, reference):
            return True

    return False


def is_builtin(identifier: AstIdentifier):
    if identifier.value.startswith("_"):
        return None

    return (
        getattr(builtins, identifier.value)
        if hasattr(builtins, identifier.value)
        else None
    )


def annotate_types(annotation):
    if inspect.isclass(annotation):
        return type[annotation]
    elif inspect.isfunction(annotation) or inspect.isbuiltin(annotation):
        return annotation
    else:
        return type(annotation)


def search_scope_for_binding(
    var_name: str, node: AstNode, scope: LexicalScope
) -> tuple[Binding, LexicalScope] | None:
    variables = scope.variables
    if var_name in variables:
        var_data = variables[var_name]

        for binding in var_data.bindings:
            if was_referenced(binding.references, node) or are_equal(
                binding.origin, node
            ):
                return (binding, scope)

    for child in scope.children:
        if binding := search_scope_for_binding(var_name, node, child):
            return binding

    return None


def get_referenced_type(
    runtime: Runtime, module: CompiledModule, identifier: AstIdentifier
):
    var_name = identifier.value
    scope = module.lexical_scope

    if binding := search_scope_for_binding(var_name, identifier, scope):
        # logging.debug(binding)
        annotation = get_type_annotation(binding[0].origin)
        # logging.debug(annotation)
        return annotation
    elif identifier.value in module.globals:
        return annotate_types(runtime.globals[identifier.value])

    elif annotation := is_builtin(identifier):
        return annotate_types(annotation)

    return UNKNOWN_TYPE


def add_representation(arg_node: AstNode, type: Any):
    metadata = retrieve_metadata(arg_node, ResourceLocationMetadata)

    if metadata is None:
        metadata = ResourceLocationMetadata()

    metadata.represents = type

    attach_metadata(arg_node, metadata)


def get_type_annotation(node: AstNode) -> Any:
    if node is None:
        return None

    metadata = retrieve_metadata(node, VariableMetadata)

    if metadata is None:
        return None

    return metadata.type_annotation


def set_type_annotation(node: AstNode, value: Any):
    metadata = retrieve_metadata(node, VariableMetadata) or VariableMetadata()

    metadata.type_annotation = value

    attach_metadata(node, metadata)


@dataclass
class InitialStep(Reducer):
    helpers: dict[str, Any] = extra_field(default_factory=dict)

    @rule(AstFromImport)
    def from_import(self, from_import: AstFromImport):
        module_path = from_import.arguments[0]

        if not isinstance(module_path, AstResourceLocation):
            return

        if module_path.namespace:
            if (
                not (
                    compilation := COMPILATION_RESULTS.get(
                        module_path.get_canonical_value()
                    )
                )
                or compilation.compiled_module is None
            ):
                return

            scope = compilation.compiled_module.lexical_scope
            # logging.debug(compilation.compiled_module)

            for argument in from_import.arguments[1:]:
                if isinstance(argument, AstImportedItem) and (
                    export := scope.variables.get(argument.name)
                ):
                    set_type_annotation(
                        argument, get_type_annotation(export.bindings[0].origin)
                    )
        else:
            self.handle_python_module(from_import, module_path)

    def handle_python_module(
        self, from_import: AstFromImport, module_path: AstResourceLocation
    ):
        try:
            module: ModuleType = self.helpers["import_module"](module_path.get_value())
        except:
            logging.error(f"Can't import module {module_path}")
            return

        for argument in from_import.arguments[1:]:
            if isinstance(argument, AstImportedItem) and hasattr(module, argument.name):
                annotation = annotate_types(getattr(module, argument.name))
                set_type_annotation(argument, annotation)

    @rule(AstValue)
    def value(self, value: AstValue):
        if get_type_annotation(value):
            return value

        set_type_annotation(value, annotate_types(value.value))

        return value

    @rule(AstResourceLocation)
    def resource_location(self, node: AstResourceLocation):

        if isinstance(node, AstNestedLocation):
            metadata = (
                retrieve_metadata(node, ResourceLocationMetadata)
                or ResourceLocationMetadata()
            )

            metadata.unresolved_path = f"~/" + node.path

            attach_metadata(node, metadata)

@dataclass
class BindingStep(Reducer):
    index: AegisProjectIndex = required_field()
    source_path: str = required_field()
    runtime: Runtime = required_field()
    mecha: Mecha = required_field()

    parser_to_file_type: dict[str, type[NamespaceFile]] = required_field()

    module: Optional[CompiledModule] = required_field()

    defined_files = []

    @rule(AstAssignment)
    def assignment(self, node: AstAssignment):
        if node.target is None:
            return node

        if node.type_annotation:
            type_annotation = node_to_types(node.type_annotation)
        else:
            expression = node.value
            type_annotation = get_type_annotation(expression)

        if type_annotation is not None:
            set_type_annotation(node.target, type_annotation)
        return node

    @rule(AstTypeDeclaration)
    def type_declaration(self, node: AstTypeDeclaration):
        annotation = node_to_types(node.type_annotation)

        set_type_annotation(node.identifier, annotation)

    @rule(AstIdentifier)
    def identifier(self, identifier):
        if (
            get_type_annotation(identifier)
            or self.module is None
            or self.runtime is None
        ):
            return identifier

        set_type_annotation(
            identifier, get_referenced_type(self.runtime, self.module, identifier)
        )

        return identifier

    @rule(AstAttribute)
    def attribute(self, attribute: AstAttribute):
        if get_type_annotation(attribute):
            return

        base = get_type_annotation(attribute.value)

        if base is UNKNOWN_TYPE:
            set_type_annotation(attribute, UNKNOWN_TYPE)
            return

        info = get_type_info(base) if not isinstance(base, TypeInfo) else base

        if attribute.name in info.fields:
            set_type_annotation(attribute, info.fields[attribute.name])
        elif attribute.name in info.functions:
            set_type_annotation(attribute, info.functions[attribute.name])
        else:
            set_type_annotation(attribute, UNKNOWN_TYPE)

        return attribute

    @rule(AstCall)
    def call(self, call: AstCall):
        if get_type_annotation(call):
            return call

        callable = get_type_annotation(call.value)

        if callable is UNKNOWN_TYPE:
            set_type_annotation(call, UNKNOWN_TYPE)
            return call

        # If the callable is a type of a type
        # then its type annotation should be the
        # method signature of its constructor
        if get_origin(callable) is type:
            callable = get_args(callable)[0]
            info = FunctionInfo.extract(callable.__init__)
            info.return_annotation = callable
        elif isinstance(callable, TypeInfo):
            info = copy(callable.functions.get("__init__")) or FunctionInfo(
                [("self", ParameterInfo(inspect._empty, inspect._empty))],
                callable,
                callable.doc,
            )
            info.return_annotation = callable
        elif isinstance(callable, FunctionInfo):
            info = callable
        else:
            info = FunctionInfo.extract(callable)

        if info is None or info.return_annotation is inspect.Parameter.empty:
            set_type_annotation(call, UNKNOWN_TYPE)
            return call

        set_type_annotation(call, info.return_annotation)
        return call

    @rule(AstCommand)
    def command(self, command: AstCommand):
        if not (prototype := self.mecha.spec.prototypes.get(command.identifier)):
            return
        
        nested_root_found = False

        for i, argument in enumerate(command.arguments):

            match argument:
                case AstResourceLocation():

                    # Attempt to get the parser for the argument
                    argument_tree = prototype.get_argument(i)
                    command_tree_node = self.mecha.spec.tree.get(argument_tree.scope)
                    if not (command_tree_node and command_tree_node.parser):
                        continue

                    # If the parser is registered or the parent argument's name is registered
                    # use that file type for its representation
                    file_type = self.parser_to_file_type.get(
                        command_tree_node.parser
                    ) or self.index.resource_name_to_type.get(argument_tree.scope[-2])

                    if file_type is None:
                        continue

                    # Ensure that unfinished paths are not added to the project index
                    resolved_path = argument.get_canonical_value()

                    # If the argument is a tag then we need to remove the leading
                    # "#" and try to change the file type to the tag equivalent
                    # ex. Function -> FunctionTag
                    if argument.is_tag:
                        resolved_path = resolved_path[1:]
                        if not (
                            file_type := self.index.resource_name_to_type.get(
                                file_type.snake_name + "_tag"
                            )
                        ):
                            continue

                    add_representation(argument, file_type)

                    if not valid_resource_location(resolved_path):
                        continue

                    # Check the command tree for the pattern:
                    # resource_location, defintion
                    # which is used by the nested resource plugin to define a new resource
                    if isinstance(
                        command.arguments[-1], (AstRoot, AstJson)
                    ) and not nested_root_found:
                        nested_root_found = True
                        self.index[file_type].add_definition(
                            resolved_path,
                            self.source_path,
                            (argument.location, argument.end_location),
                        )
                    # If the pattern isn't matched then just treat it as a reference
                    # and not a definition of thre resource
                    else:
                        self.index[file_type].add_reference(
                            resolved_path,
                            self.source_path,
                            (argument.location, argument.end_location),
                        )

    @rule(AstBlock)
    def block(self, block: AstBlock):
        add_representation(block.identifier, "block")

    @rule(AstItemStack)
    def item_stack(self, item_stack: AstItemStack):
        add_representation(item_stack.identifier, "block")

    @rule(AstParticle)
    def particle(self, particle: AstParticle):
        add_representation(particle.name, "particle_type")

    @rule(AstSelectorArgument)
    def selector_argument(self, selector_argument: AstSelectorArgument):
        key = selector_argument.key.value

        if not isinstance(selector_argument.value, AstResourceLocation):
            return

        value = selector_argument.value

        match key:
            case "type":
                add_representation(value, "entity_type")
            case "predicate":
                add_representation(value, Predicate)

    def add_field(self, type_info: TypeInfo, node: AstNode):
        def add_target_identifier(target: AstTargetIdentifier):
            name = target.value
            annotation = get_type_annotation(target)

            if name in type_info.fields:
                type_info.fields[name] = type_info.fields[name] | annotation
            else:
                type_info.fields[name] = annotation

        if isinstance(node, AstAssignment) and isinstance(
            node.target, (AstTargetIdentifier)
        ):
            add_target_identifier(node.target)

        elif isinstance(node, AstTypeDeclaration):
            add_target_identifier(node.identifier)

    @rule(AstCommand, identifier="class:name:bases:body")
    def command_class_body(self, command: AstCommand):
        name = cast(AstClassName, command.arguments[0])

        type_info = get_type_annotation(name)
        if type_info:
            return

        body = cast(AstRoot, command.arguments[2])

        doc = None
        if len(body.commands) > 0 and isinstance(body.commands[0], AstDocstring):
            doc = cast(AstCommand, body.commands[0])
            value = cast(AstValue, doc.arguments[0])

            doc = value.value

        type_info = TypeInfo(doc)

        for c in body.commands:
            if isinstance(c, AstError):
                continue

            if isinstance(c, AstStatement):
                self.add_field(type_info, c.arguments[0])
            elif c.identifier == "def:function:body":
                signature = cast(AstFunctionSignature, c.arguments[0])
                annotation = get_type_annotation(signature)

                if not isinstance(annotation, FunctionInfo):
                    continue

                type_info.functions[signature.name] = annotation

        set_type_annotation(name, type_info)

    @rule(AstCommand, identifier="def:function:body")
    def command_function_body(self, command: AstCommand):
        signature = cast(AstFunctionSignature, command.arguments[0])

        metadata = retrieve_metadata(signature, VariableMetadata)

        function_info = metadata.type_annotation if metadata else None
        if not function_info or not isinstance(function_info, FunctionInfo):
            return

        body = cast(AstRoot, command.arguments[1])

        if len(body.commands) > 0 and isinstance(body.commands[0], AstDocstring):
            doc = cast(AstCommand, body.commands[0])
            value = cast(AstValue, doc.arguments[0])

            function_info.doc = value.value

    @rule(AstFunctionSignature)
    def function_signature(self, signature: AstFunctionSignature):
        arguments = []

        for argument in signature.arguments:

            if not isinstance(
                argument,
                (
                    AstFunctionSignatureArgument,
                    AstFunctionSignatureVariadicArgument,
                    AstFunctionSignatureVariadicKeywordArgument,
                ),
            ):
                continue

            annotation = (
                node_to_types(argument.type_annotation)
                if argument.type_annotation
                else inspect._empty
            )

            match argument:
                case AstFunctionSignatureArgument():
                    position = inspect.Parameter.POSITIONAL_OR_KEYWORD
                case AstFunctionSignatureVariadicArgument():
                    position = inspect.Parameter.VAR_POSITIONAL
                case AstFunctionSignatureVariadicKeywordArgument():
                    position = inspect.Parameter.VAR_KEYWORD
                case _:
                    position = inspect.Parameter.POSITIONAL_ONLY

            arguments.append(
                inspect.Parameter(
                    argument.name,
                    position,
                    annotation=annotation,
                )
            )

        return_type: Any = (
            node_to_types(signature.return_type_annotation)
            if signature.return_type_annotation
            else UNKNOWN_TYPE
        )

        annotation = FunctionInfo.from_signature(
            inspect.Signature(arguments, return_annotation=return_type),
            None,
            dict(),
        )

        set_type_annotation(
            signature,
            annotation,
        )

        # logging.debug("Scanning references")
        # logging.debug(self.module)
        if self.module is None:
            return

        # logging.debug("get scope")
        if result := search_scope_for_binding(
            signature.name, signature, self.module.lexical_scope
        ):
            # logging.debug(result)
            for reference in result[0].references:
                if get_type_annotation(reference) is None:
                    set_type_annotation(reference, annotation)
        # logging.debug("done")


@dataclass
class Indexer(MutatingReducer):
    ctx: LanguageServerContext = required_field()
    resource_location: str = required_field()
    source_path: str = required_field()
    file_instance: Function | Module = required_field()

    output_ast: AstRoot = extra_field(
        default=AstRoot(commands=AstChildren(children=[]))
    )

    def __call__(self, ast: AstRoot, *args) -> AbstractNode:
        project_index = self.ctx.inject(AegisProjectIndex)

        mecha = self.ctx.inject(Mecha)
        runtime = self.ctx.inject(Runtime)
        module = runtime.modules[self.file_instance]
        # logging.debug(id(ast))
        ast = module.ast if module is not None else ast
        # logging.debug(id(ast))

        # A file always defines itself
        source_type = type(self.file_instance)

        project_index[source_type].add_definition(
            self.resource_location, self.source_path
        )

        # TODO: See if these steps can be merged into one

        # Attaches the type annotations for assignments
        initial_values = InitialStep(helpers=runtime.helpers)

        # The binding step is responsible for attaching the majority of type annotations
        bindings = BindingStep(
            index=project_index,
            source_path=self.source_path,
            module=module,
            runtime=self.ctx.inject(Runtime),
            mecha=self.ctx.inject(Mecha),
            # argument parser to resource type
            parser_to_file_type={
                "minecraft:advancement": Advancement,
                "minecraft:function": Function,
                "minecraft:predicate": Predicate,
                "minecraft:loot_table": LootTable,
            },
        )

        # This has to been done through extension because i'm too lazy to shadow or patch it
        self.extend(
            NestedLocationTransformer(
                nested_location_resolver=NestedLocationResolver(ctx=self.ctx)
            )
        )

        steps: list[Callable[[AstRoot], AstRoot]] = [
            initial_values,
            super().__call__,
            bindings,
        ]

        for step in steps:
            try:
                ast = step(ast)
            except CompilationError as e:
                raise e.__cause__
            except Exception as e:
                tb = "\n".join(traceback.format_tb(e.__traceback__))
                logging.error(f"Error occured during {step}\n{e}\n{tb}")

        self.output_ast = ast

        # Return a deepcopy so subsequent compilation steps don't modify the parsed state
        return deepcopy(ast)
