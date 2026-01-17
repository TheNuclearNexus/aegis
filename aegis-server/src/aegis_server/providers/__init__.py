from aegis_core.ast.features import AegisFeatureProviders
from aegis_core.ast.features.provider import BaseFeatureProvider
from beet import Context
from bolt import (
    AstAttribute,
    AstClassName,
    AstFormatString,
    AstFunctionSignature,
    AstFunctionSignatureArgument,
    AstIdentifier,
    AstImportedItem,
    AstTargetAttribute,
    AstTargetIdentifier,
    AstValue,
)
from mecha import AstItemSlot, AstNode, AstResourceLocation

from .resource_location import *
from .variable import *
from .generic import *

__all__ = [
    "register_providers"
]

PROVIDERS: dict[type[AstNode], type[BaseFeatureProvider]] = {
    # Bolt Specific
    AstIdentifier: VariableFeatureProvider,
    AstAttribute: VariableFeatureProvider,
    AstTargetAttribute: VariableFeatureProvider,
    AstTargetIdentifier: VariableFeatureProvider,
    AstImportedItem: VariableFeatureProvider,
    AstClassName: ClassNameProvider,
    AstFunctionSignature: FunctionSignatureProvider,
    AstFunctionSignatureArgument: FunctionSignatureArgProvider,
    AstValue: ValueProvider,
    AstFormatString: FormatStringProvider,
    # Mecha Specific
    AstResourceLocation: ResourceLocationFeatureProvider,
    AstItemSlot: ItemSlotProvider,
}


def register_providers(ctx: Context):
    providers = ctx.inject(AegisFeatureProviders)

    for node, provider in PROVIDERS.items():
        providers.attach(node, provider)

