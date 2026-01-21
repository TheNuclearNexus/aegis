from contextlib import contextmanager
import logging
from pathlib import Path
import traceback
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from types import ModuleType
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

from aegis_core.ast.helpers import _hash_node
from beet import (
    Advancement,
    Function,
    LootTable,
    NamespaceFile,
    Predicate,
    TextFileBase,
)
from beet.core.utils import extra_field, required_field
from bolt import (
    AstAssignment,
    AstAttribute,
    AstCall,
    AstClassBases,
    AstClassName,
    AstClassRoot,
    AstDict,
    AstDocstring,
    AstExpression,
    AstExpressionBinary,
    AstFromImport,
    AstFunctionSignature,
    AstFunctionSignatureArgument,
    AstFunctionSignatureElement,
    AstFunctionSignatureVariadicArgument,
    AstFunctionSignatureVariadicKeywordArgument,
    AstIdentifier,
    AstImportedItem,
    AstList,
    AstLookup,
    AstStatement,
    AstTargetIdentifier,
    AstTuple,
    AstTypeAnnotation,
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
    clear_metadata,
    retrieve_metadata,
)
from aegis_core.indexing.project_index import AegisProjectIndex, valid_resource_location
from aegis_core.reflection import (
    UNKNOWN_TYPE,
    FunctionInfo,
    ParameterInfo,
    TypeInfo,
    annotate_types,
    get_type_info,
    is_builtin,
    search_scope_for_binding,
)
from aegis_core.reflection.type_representation import (
    ClassRepresentation,
    BuiltinRepresentation,
    ModuleRepresentation,
    ReferencedTypeRepresentation,
    TypeRepresentation,
    UnionRepresentation,
    InstanceRepresentation,
    FunctionRepresentation,
    ParameterRepresentation,
)

from ..shadows.compile_document import COMPILATION_RESULTS
from ..shadows.context import LanguageServerContext

Node = TypeVar("Node", bound=AstNode)


# def node_to_types(node: AstNode):

#     types = []
#     for n in node.walk():
#         if isinstance(n, AstExpressionBinary) and n.operator == "|":
#             continue

#         annotation = expression_to_annotation(n)

#         if annotation is not UNKNOWN_TYPE:
#             types.append(annotation)

#     if len(types) == 0:
#         return UNKNOWN_TYPE
#     elif len(types) == 1:
#         return types[0]

#     # TODO: Fix type unions
#     return []
#     return reduce(lambda a, b: a | b, types)


# def expression_to_annotation(expression):
#     type_annotation = UNKNOWN_TYPE
#     match (expression):
#         case AstValue() as value:
#             type_annotation = type(value.value)
#         case AstDict() as _dict:
#             type_annotation = dict
#         case AstList():
#             type_annotation = list
#         case AstTuple():
#             type_annotation = tuple
#         case AstIdentifier():
#             identifier_type = get_type_annotation(expression)
#             if get_origin(identifier_type) is not type:
#                 return UNKNOWN_TYPE

#             type_annotation = get_args(identifier_type)[0]
#     return type_annotation


# def get_referenced_type(
#     runtime: Runtime, module: CompiledModule, identifier: AstIdentifier
# ) -> TypeRepresentation:
#     var_name = identifier.value
#     scope = module.lexical_scope

#     if binding := search_scope_for_binding(var_name, identifier, scope):
#         annotation = get_type_annotation(binding[0].origin)
#         return annotation or UNKNOWN_TYPE

#     elif identifier.value in module.globals:
#         global_var = runtime.globals[identifier.value]
#         return TypeRepresentation.from_python(global_var)

#     elif annotation := is_builtin(identifier):
#         return TypeRepresentation.from_python(annotation)

#     return UNKNOWN_TYPE


def add_representation(resource_location, arg_node: AstNode, type: Any):
    metadata = retrieve_metadata(resource_location, arg_node, ResourceLocationMetadata)

    if metadata is None:
        metadata = ResourceLocationMetadata()

    metadata.represents = type

    attach_metadata(resource_location, arg_node, metadata)


@dataclass
class InitialStep(Reducer):
    helpers: dict[str, Any] = extra_field(default_factory=dict)

    # @rule(AstFromImport)
    # def from_import(self, from_import: AstFromImport):
    #     module_path = from_import.arguments[0]

    #     if not isinstance(module_path, AstResourceLocation):
    #         return from_import

    #     if module_path.namespace:
    #         if (
    #             not (
    #                 compilation := COMPILATION_RESULTS.get(
    #                     module_path.get_canonical_value()
    #                 )
    #             )
    #             or compilation.compiled_module is None
    #         ):
    #             return from_import

    #         scope = compilation.compiled_module.lexical_scope
    #         # logging.debug(compilation.compiled_module)

    #         for argument in from_import.arguments[1:]:
    #             if isinstance(argument, AstImportedItem) and (
    #                 export := scope.variables.get(argument.name)
    #             ):
    #                 set_type_annotation(
    #                     argument, get_type_annotation(export.bindings[0].origin)
    #                 )
    #     else:
    #         self.handle_python_module(from_import, module_path)

    #     return from_import

    # def handle_python_module(
    #     self, from_import: AstFromImport, module_path: AstResourceLocation
    # ):
    #     try:
    #         module: ModuleType = self.helpers["import_module"](module_path.get_value())
    #     except:
    #         logging.error(f"Can't import module {module_path}")
    #         return

    #     for argument in from_import.arguments[1:]:
    #         if isinstance(argument, AstImportedItem) and hasattr(module, argument.name):
    #             annotation = TypeRepresentation.from_python(getattr(module, argument.name))
    #             set_type_annotation(argument, annotation)

    # @rule(AstValue)
    # def value(self, value: AstValue):
    #     if has_type_annotation(value):
    #         return value

    #     set_type_annotation(value, TypeRepresentation.from_node(value.value))

    #     return value

    # @rule(AstResourceLocation)
    # def resource_location(self, node: AstResourceLocation):

    #     if isinstance(node, AstNestedLocation):
    #         metadata = (
    #             retrieve_metadata(node, ResourceLocationMetadata)
    #             or ResourceLocationMetadata()
    #         )

    #         metadata.unresolved_path = f"~/" + node.path

    #         attach_metadata(node, metadata)

    #     return node


type IdentifierLike = AstIdentifier | AstFunctionSignatureArgument


@dataclass
class BindingStep(Reducer):
    ctx: LanguageServerContext = required_field()
    index: AegisProjectIndex = required_field()
    source_path: str = required_field()
    resource_location: str = required_field()
    runtime: Runtime = required_field()
    mecha: Mecha = required_field()

    parser_to_file_type: dict[str, type[NamespaceFile]] = required_field()

    module: Optional[CompiledModule] = required_field()

    defined_files = []

    def type_analysis(self, node: AstNode):
        if self.has_type_annotation(node):
            return False

        self.set_type_annotation(
            node, ReferencedTypeRepresentation.from_node(self.resource_location, node)
        )
        return True

    def has_type_annotation(self, node: AstNode) -> bool:
        if node is None:
            return True

        if retrieve_metadata(self.resource_location, node, VariableMetadata):
            return True

        return False

    def get_type_annotation(self, node: AstNode) -> TypeRepresentation:
        if node is None:
            return UNKNOWN_TYPE

        if self.has_type_annotation(node):
            metadata = retrieve_metadata(self.resource_location, node, VariableMetadata)

            if metadata is None:
                return UNKNOWN_TYPE

            annotation = metadata.type_annotation
            tree = [annotation]
            while isinstance(annotation, ReferencedTypeRepresentation):
                metadata = retrieve_metadata(
                    annotation.resource_location, annotation.hash, VariableMetadata
                )
                assert metadata
                annotation = metadata.type_annotation
                if annotation in tree:
                    annotation = UNKNOWN_TYPE
                    break

                tree.append(annotation)

            return annotation

        # Run the binding step on the node to get it's type annotation
        self.invoke(node)
        metadata = retrieve_metadata(self.resource_location, node, VariableMetadata)
        return metadata.type_annotation if metadata else UNKNOWN_TYPE

    def get_type_annotation_ref(self, node: AstNode) -> TypeRepresentation:
        annotation = self.get_type_annotation(node)

        if annotation is UNKNOWN_TYPE:
            return UNKNOWN_TYPE

        if isinstance(annotation, ReferencedTypeRepresentation):
            return annotation

        return ReferencedTypeRepresentation.from_node(self.resource_location, node)

    def set_type_annotation(self, node: AstNode, value: TypeRepresentation):
        metadata = (
            retrieve_metadata(self.resource_location, node, VariableMetadata)
            or VariableMetadata()
        )

        metadata.type_annotation = value

        attach_metadata(self.resource_location, node, metadata)

    @rule(AstValue)
    def value(self, node: AstValue):
        if self.type_analysis(node):
            annotation = TypeRepresentation.from_python(type(node.value))

            self.set_type_annotation(node, InstanceRepresentation(annotation))

        return node

    @rule(AstCommand, identifier="def:function:body")
    def function_body(self, node: AstCommand):
        if not self.has_type_annotation(node.arguments[0]):
            return node

        signature = cast(AstFunctionSignature, node.arguments[0])
        body = cast(AstRoot, node.arguments[1])

        doc_string = None

        # TODO: anaylze python returns to get return type
        for command in body.commands:
            if isinstance(command, AstDocstring):
                doc_string = cast(str, cast(AstValue, command.arguments[0]).value)
                break

        annotation = self.get_type_annotation(signature)

        if isinstance(annotation, FunctionRepresentation):
            logging.debug(f"\n\n{id(annotation)}\n{_hash_node(signature)}\n\n")
            annotation.doc_string = doc_string

        return node

    @rule(AstFunctionSignature)
    def function_sig(self, node: AstFunctionSignature):
        if self.type_analysis(node):
            arguments = []
            var_arguments = []
            var_kw_arguments = []

            for arg in node.arguments:
                match arg:
                    case AstFunctionSignatureArgument():
                        annotation = self.get_type_annotation(arg)
                        if isinstance(annotation, InstanceRepresentation):
                            annotation = annotation.parent

                        arguments.append(
                            ParameterRepresentation(arg.name, annotation, arg.default)
                        )
                    case AstFunctionSignatureVariadicArgument():
                        annotation = self.get_type_annotation(arg)
                        if isinstance(annotation, InstanceRepresentation):
                            annotation = annotation.parent

                        var_arguments.append(
                            ParameterRepresentation(arg.name, annotation, None)
                        )
                    case AstFunctionSignatureVariadicKeywordArgument():
                        annotation = self.get_type_annotation(arg)
                        if isinstance(annotation, InstanceRepresentation):
                            annotation = annotation.parent

                        var_kw_arguments.append(
                            ParameterRepresentation(arg.name, annotation, None)
                        )

            return_type = (
                self.get_type_annotation(node.return_type_annotation)
                if node.return_type_annotation
                else UNKNOWN_TYPE
            )

            self.set_type_annotation(
                node,
                FunctionRepresentation(
                    node.name,
                    doc_string=None,
                    arguments=arguments,
                    var_arguments=var_arguments,
                    var_kw_arguments=var_kw_arguments,
                    return_type=return_type,
                ),
            )

        return node

    @rule(AstFunctionSignatureArgument)
    def function_sig_arg(self, node: AstFunctionSignatureArgument):
        if self.type_analysis(node):
            if node.type_annotation and (
                annotation := self.get_type_annotation(node.type_annotation)
            ):
                self.set_type_annotation(node, InstanceRepresentation(annotation))
                return node

            if node.default and (annotation := self.get_type_annotation(node.default)):
                self.set_type_annotation(node, annotation)
                return node

            self.set_type_annotation(node, UNKNOWN_TYPE)

        return node

    @rule(AstFunctionSignatureVariadicArgument)
    def function_sig_var_arg(self, node: AstFunctionSignatureVariadicArgument):
        if self.type_analysis(node):
            if node.type_annotation and (
                annotation := self.get_type_annotation(node.type_annotation)
            ):
                self.set_type_annotation(node, InstanceRepresentation(annotation))
                return node

            self.set_type_annotation(node, UNKNOWN_TYPE)

        return node

    @rule(AstFunctionSignatureVariadicKeywordArgument)
    def function_sig_kw_arg(self, node: AstFunctionSignatureVariadicKeywordArgument):
        if self.type_analysis(node):
            if node.type_annotation and (
                annotation := self.get_type_annotation(node.type_annotation)
            ):
                self.set_type_annotation(node, InstanceRepresentation(annotation))
                return node

            self.set_type_annotation(node, UNKNOWN_TYPE)

        return node

    @rule(AstIdentifier)
    def identifier(self, node: AstIdentifier):
        if self.type_analysis(node):
            if self.module:
                if binding := search_scope_for_binding(
                    node.value, node, self.module.lexical_scope
                ):
                    annotation = self.get_type_annotation_ref(binding[0].origin)
                    self.set_type_annotation(node, annotation)
                    return node

            if builtin := is_builtin(node):
                self.set_type_annotation(node, TypeRepresentation.from_python(builtin))
            else:
                self.set_type_annotation(node, UNKNOWN_TYPE)

        return node

    @rule(AstCall)
    def call(self, node: AstCall):
        if self.type_analysis(node):

            annotation = self.get_type_annotation(node.value)

            if isinstance(annotation, InstanceRepresentation):
                annotation = annotation.parent

            match annotation:
                case ClassRepresentation() as cls:
                    self.set_type_annotation(node, InstanceRepresentation(cls))
                case FunctionRepresentation() as func:
                    self.set_type_annotation(node, func.return_type)

        return node

    @rule(AstLookup)
    def lookup(self, node: AstLookup):
        if self.type_analysis(node):
            base_type = self.get_type_annotation(node.value)

            if not base_type:
                self.set_type_annotation(node, UNKNOWN_TYPE)
                return node

            if isinstance(base_type, ClassRepresentation):
                arguments = [self.get_type_annotation(n) for n in node.arguments]

                base_type = ClassRepresentation(
                    name=base_type.name,
                    fields=base_type.fields,
                    methods=base_type.methods,
                    doc_string=base_type.doc_string,
                    generics=arguments,
                )
                self.set_type_annotation(node, base_type)
            else:
                self.set_type_annotation(node, base_type)

        return arguments

    @rule(AstTypeAnnotation)
    def annotation(self, node: AstTypeAnnotation):
        if self.type_analysis(node):
            types = []
            for child in node:
                types.append(self.get_type_annotation(child))

            if len(types) == 1:
                self.set_type_annotation(node, types[0])
            elif len(types) > 1:
                self.set_type_annotation(node, UnionRepresentation(types))
            else:
                self.set_type_annotation(node, UNKNOWN_TYPE)

        return node

    @rule(AstTypeDeclaration)
    def type_declaration(self, node: AstTypeDeclaration):
        if self.type_analysis(node):
            annotation = self.get_type_annotation(node.type_annotation)

            self.set_type_annotation(node, InstanceRepresentation(annotation))

        return node

    @rule(AstAssignment)
    def assignment(self, node: AstAssignment):
        if self.type_analysis(node.target):
            if node.type_annotation:
                annotation = self.get_type_annotation(node.type_annotation)
                if annotation:
                    self.set_type_annotation(
                        node.target, InstanceRepresentation(annotation)
                    )
                    return node

            value_type = self.get_type_annotation(node.value)

            self.set_type_annotation(node.target, value_type)

        return node

    @rule(AstAttribute)
    def attribute(self, node: AstAttribute):
        if self.type_analysis(node):
            base_type = self.get_type_annotation(node.value)

            if isinstance(base_type, InstanceRepresentation):
                base_type = base_type.parent

            match base_type:
                case ClassRepresentation():
                    if field := base_type.get_field(node.name):
                        self.set_type_annotation(node, InstanceRepresentation(field[1]))
                        return node

                    if method := base_type.get_method(node.name):
                        self.set_type_annotation(node, method[1])
                        return node

            self.set_type_annotation(node, UNKNOWN_TYPE)
        return node

    @rule(AstCommand, identifier="class:name:body")
    def class_name_body(self, node: AstCommand):
        if self.type_analysis(node):
            return self.handle_class_defs(node)
        return node

    @rule(AstCommand, identifier="class:name:bases:body")
    def class_name_bases_body(self, node: AstCommand):
        if self.type_analysis(node):
            return self.handle_class_defs(node)
        return node

    def handle_class_defs(self, node: AstCommand):
        name = cast(AstClassName, node.arguments[0])

        if node.identifier == "class:name:bases:body":
            bases: Sequence[AstExpression] = cast(
                AstClassBases, node.arguments[1]
            ).inherit
            body = cast(AstClassRoot, node.arguments[2])
        else:
            bases = []
            body = cast(AstClassRoot, node.arguments[1])

        base_types = []
        for base in bases:
            base_types.append(self.get_type_annotation(base))

        doc_string = None
        fields: list[tuple[str, TypeRepresentation, Any]] = []
        methods: list[tuple[str, TypeRepresentation]] = []

        for command in body.commands:
            if isinstance(command, AstError):
                continue

            if doc_string is None and isinstance(command, AstDocstring):
                doc_string = cast(str, cast(AstValue, command.arguments[0]).value)

            if command.identifier == "def:function:body":
                signature = cast(AstFunctionSignature, command.arguments[0])
                annotation = self.get_type_annotation_ref(signature)

                methods.append((signature.name, annotation))

            if isinstance(command, AstStatement):
                command = command.arguments[0]

                if isinstance(command, AstAssignment) and isinstance(
                    command.target, AstTargetIdentifier
                ):
                    annotation = self.get_type_annotation(command.target)
                    if isinstance(annotation, InstanceRepresentation):
                        annotation = annotation.parent

                    fields.append((command.target.value, annotation, command.value))

                if isinstance(command, AstTypeDeclaration):
                    annotation = self.get_type_annotation(command.type_annotation)

                    fields.append((command.identifier.value, annotation, None))

        class_repr = ClassRepresentation(
            name=name.value,
            doc_string=doc_string,
            fields=fields,
            methods=methods,
            generics=[],
            bases=base_types,
        )

        self.set_type_annotation(node, class_repr)
        self.set_type_annotation(name, class_repr)

        return node

    def get_bolt_module_type(self, resource_location: AstResourceLocation):
        module = UNKNOWN_TYPE
        path = resource_location.get_canonical_value()
        for _type in [Function, Module]:
            file = self.ctx.data[cast(type[NamespaceFile], _type)].get(path)
            if file is None:
                continue

            module = self.runtime.modules.get(cast(TextFileBase[Any], file))

            assert module, "Referenced module has not been compiled"

            if not module.ast:
                continue

            metadata = retrieve_metadata(path, module.ast, VariableMetadata)

            if not metadata:
                BindingStep(
                    ctx=self.ctx,
                    index=self.index,
                    resource_location=path,
                    source_path=str(Path(file.ensure_source_path()).absolute()),
                    module=module,
                    mecha=self.mecha,
                    runtime=self.runtime,
                    parser_to_file_type=self.parser_to_file_type
                )(module.ast)

                metadata = retrieve_metadata(path, module.ast, VariableMetadata)
                if not metadata:
                    logging.warning(f"Attempted to index file {path} but failed.")
                    return UNKNOWN_TYPE

            annotation = metadata.type_annotation

            if annotation is not None and isinstance(annotation, ModuleRepresentation):
                module = annotation
                break
        return module

    @rule(AstFromImport)
    def from_import(self, node: AstFromImport):
        if self.type_analysis(node):
            path = cast(AstResourceLocation, node.arguments[0])

            if path.namespace:
                module = self.get_bolt_module_type(path)
                if module is UNKNOWN_TYPE or not isinstance(
                    module, ModuleRepresentation
                ):
                    self.set_type_annotation(node, UNKNOWN_TYPE)

                    for n in node.arguments[1:]:
                        self.set_type_annotation(n, UNKNOWN_TYPE)

                    return node

                self.set_type_annotation(node, module)
                for n in node.arguments[1:]:
                    match n:
                        case AstImportedItem():
                            annotation = module.get_member(n.name)
                            self.set_type_annotation(n, annotation or UNKNOWN_TYPE)
        return node

    @rule(AstRoot)
    def root(self, node: AstRoot):
        if node.location.pos != 0:
            return node

        if self.type_analysis(node):
            if not self.module:
                self.set_type_annotation(node, UNKNOWN_TYPE)
                return node

            members = []

            for name, variable in self.module.lexical_scope.variables.items():
                types = []
                for binding in variable.bindings:
                    annotation = self.get_type_annotation(binding.origin)
                    types.append(annotation)
                    # ...

                if len(types) == 1:
                    members.append((name, types[0]))
                elif len(types) > 1:
                    members.append((name, UnionRepresentation(types)))
                else:
                    members.append((name, UNKNOWN_TYPE))

            doc_string = None

            for command in node.commands:
                if isinstance(command, AstError):
                    continue

                if isinstance(command, AstDocstring):
                    doc_string = cast(str, cast(AstValue, command.arguments[0]).value)
                    break

            self.set_type_annotation(node, ModuleRepresentation(doc_string, members))
        return node

    # @rule(AstCommand)
    # def command(self, command: AstCommand):
    #     if not (prototype := self.mecha.spec.prototypes.get(command.identifier)):
    #         return command

    #     nested_root_found = False

    #     for i, argument in enumerate(command.arguments):

    #         match argument:
    #             case AstResourceLocation():

    #                 # Attempt to get the parser for the argument
    #                 argument_tree = prototype.get_argument(i)
    #                 command_tree_node = self.mecha.spec.tree.get(argument_tree.scope)
    #                 if not (command_tree_node and command_tree_node.parser):
    #                     continue

    #                 # If the parser is registered or the parent argument's name is registered
    #                 # use that file type for its representation
    #                 file_type = self.parser_to_file_type.get(
    #                     command_tree_node.parser
    #                 ) or self.index.resource_name_to_type.get(argument_tree.scope[-2])

    #                 if file_type is None:
    #                     continue

    #                 # Ensure that unfinished paths are not added to the project index
    #                 resolved_path = argument.get_canonical_value()

    #                 # If the argument is a tag then we need to remove the leading
    #                 # "#" and try to change the file type to the tag equivalent
    #                 # ex. Function -> FunctionTag
    #                 if argument.is_tag:
    #                     resolved_path = resolved_path[1:]
    #                     if not (
    #                         file_type := self.index.resource_name_to_type.get(
    #                             file_type.snake_name + "_tag"
    #                         )
    #                     ):
    #                         continue

    #                 add_representation(argument, file_type)

    #                 if not valid_resource_location(resolved_path):
    #                     continue

    #                 # Check the command tree for the pattern:
    #                 # resource_location, defintion
    #                 # which is used by the nested resource plugin to define a new resource
    #                 if (
    #                     isinstance(command.arguments[-1], (AstRoot, AstJson))
    #                     and not nested_root_found
    #                 ):
    #                     nested_root_found = True
    #                     self.index[file_type].add_definition(
    #                         resolved_path,
    #                         self.source_path,
    #                         (argument.location, argument.end_location),
    #                     )
    #                 # If the pattern isn't matched then just treat it as a reference
    #                 # and not a definition of thre resource
    #                 else:
    #                     self.index[file_type].add_reference(
    #                         resolved_path,
    #                         self.source_path,
    #                         (argument.location, argument.end_location),
    #                     )

    #     return command

    # @rule(AstBlock)
    # def block(self, block: AstBlock):
    #     add_representation(block.identifier, "block")
    #     return block

    # @rule(AstItemStack)
    # def item_stack(self, item_stack: AstItemStack):
    #     add_representation(item_stack.identifier, "block")
    #     return item_stack

    # @rule(AstParticle)
    # def particle(self, particle: AstParticle):
    #     add_representation(particle.name, "particle_type")
    #     return particle

    # @rule(AstSelectorArgument)
    # def selector_argument(self, selector_argument: AstSelectorArgument):
    #     key = selector_argument.key.value

    #     if not isinstance(selector_argument.value, AstResourceLocation):
    #         return selector_argument

    #     value = selector_argument.value

    #     match key:
    #         case "type":
    #             add_representation(value, "entity_type")
    #         case "predicate":
    #             add_representation(value, Predicate)

    #     return selector_argument

    # def add_field(self, type_info: TypeInfo, node: AstNode):
    #     def add_target_identifier(target: AstTargetIdentifier):
    #         name = target.value
    #         annotation = get_type_annotation(target)

    #         if name in type_info.fields:
    #             type_info.fields[name] = type_info.fields[name] | annotation
    #         else:
    #             type_info.fields[name] = annotation

    #     if isinstance(node, AstAssignment) and isinstance(
    #         node.target, (AstTargetIdentifier)
    #     ):
    #         add_target_identifier(node.target)

    #     elif isinstance(node, AstTypeDeclaration):
    #         add_target_identifier(node.identifier)

    # @rule(AstCommand, identifier="class:name:bases:body")
    # def command_class_body(self, command: AstCommand):
    #     name = cast(AstClassName, command.arguments[0])

    #     type_info = get_type_annotation(name)
    #     if type_info:
    #         return command

    #     body = cast(AstRoot, command.arguments[2])

    #     doc = None
    #     if len(body.commands) > 0 and isinstance(body.commands[0], AstDocstring):
    #         doc = cast(AstCommand, body.commands[0])
    #         value = cast(AstValue, doc.arguments[0])

    #         doc = value.value

    #     type_info = TypeInfo(doc)

    #     for c in body.commands:
    #         if isinstance(c, AstError):
    #             continue

    #         if isinstance(c, AstStatement):
    #             self.add_field(type_info, c.arguments[0])
    #         elif c.identifier == "def:function:body":
    #             signature = cast(AstFunctionSignature, c.arguments[0])
    #             annotation = get_type_annotation(signature)

    #             if not isinstance(annotation, FunctionInfo):
    #                 continue

    #             type_info.functions[signature.name] = annotation

    #     set_type_annotation(name, type_info)

    #     return command

    # @rule(AstCommand, identifier="def:function:body")
    # def command_function_body(self, command: AstCommand):
    #     signature = cast(AstFunctionSignature, command.arguments[0])

    #     metadata = retrieve_metadata(signature, VariableMetadata)

    #     function_info = metadata.type_annotation if metadata else None
    #     if not function_info or not isinstance(function_info, FunctionInfo):
    #         return command

    #     body = cast(AstRoot, command.arguments[1])

    #     if len(body.commands) > 0 and isinstance(body.commands[0], AstDocstring):
    #         doc = cast(AstCommand, body.commands[0])
    #         value = cast(AstValue, doc.arguments[0])

    #         function_info.doc = value.value

    #     return command

    # @rule(AstFunctionSignature)
    # def function_signature(self, signature: AstFunctionSignature):
    #     arguments = []

    #     for argument in signature.arguments:

    #         if not isinstance(
    #             argument,
    #             (
    #                 AstFunctionSignatureArgument,
    #                 AstFunctionSignatureVariadicArgument,
    #                 AstFunctionSignatureVariadicKeywordArgument,
    #             ),
    #         ):
    #             continue

    #         annotation = (
    #             node_to_types(argument.type_annotation)
    #             if argument.type_annotation
    #             else inspect._empty
    #         )

    #         match argument:
    #             case AstFunctionSignatureArgument():
    #                 position = inspect.Parameter.POSITIONAL_OR_KEYWORD
    #             case AstFunctionSignatureVariadicArgument():
    #                 position = inspect.Parameter.VAR_POSITIONAL
    #             case AstFunctionSignatureVariadicKeywordArgument():
    #                 position = inspect.Parameter.VAR_KEYWORD
    #             case _:
    #                 position = inspect.Parameter.POSITIONAL_ONLY

    #         arguments.append(
    #             inspect.Parameter(
    #                 argument.name,
    #                 position,
    #                 annotation=annotation,
    #             )
    #         )

    #     return_type: Any = (
    #         node_to_types(signature.return_type_annotation)
    #         if signature.return_type_annotation
    #         else UNKNOWN_TYPE
    #     )

    #     annotation = FunctionInfo.from_signature(
    #         inspect.Signature(arguments, return_annotation=return_type),
    #         None,
    #         dict(),
    #     )

    #     set_type_annotation(
    #         signature,
    #         annotation,
    #     )

    #     if self.module is None:
    #         return signature

    #     if result := search_scope_for_binding(
    #         signature.name, signature, self.module.lexical_scope
    #     ):
    #         for reference in result[0].references:
    #             if get_type_annotation(reference) is None:
    #                 set_type_annotation(reference, annotation)

    #     return signature


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
            ctx=self.ctx,
            index=project_index,
            source_path=self.source_path,
            resource_location=self.resource_location,
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

        clear_metadata(self.resource_location)

        for step in steps:
            try:
                ast = step(ast)
            except CompilationError as e:
                tb = "\n".join(traceback.format_tb(e.__cause__.__traceback__))
                logging.error(f"Error occured during {step}\n{e.__cause__}\n{tb}")
                raise e.__cause__
            except Exception as e:
                tb = "\n".join(traceback.format_tb(e.__traceback__))
                logging.error(f"Error occured during {step}\n{e}\n{tb}")

        self.output_ast = ast

        # Return a deepcopy so subsequent compilation steps don't modify the parsed state
        return deepcopy(ast)
