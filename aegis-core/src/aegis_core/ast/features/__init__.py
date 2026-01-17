from dataclasses import dataclass, field

from beet import Context

from mecha import AstNode

from .provider import *


@dataclass
class AegisFeatureProviders:
    ctx: Context
    _providers: dict[type[AstNode], type[BaseFeatureProvider]] = field(
        init=False, default_factory=dict
    )

    def attach(self, node_type: type[AstNode], provider: type[BaseFeatureProvider]):
        self._providers[node_type] = provider

    def retrieve(
        self,
        node_type: type[AstNode] | AstNode,
    ) -> type[BaseFeatureProvider]:
        if not isinstance(node_type, type):
            node_type = type(node_type)

        return self._providers.get(node_type, BaseFeatureProvider)
