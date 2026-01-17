from dataclasses import dataclass, field

from beet import Context


@dataclass
class AegisGameRegistries:
    ctx: Context

    registries: dict[str, list[str]] = field(init=False, default_factory=dict)

    def __getitem__(self, registry: str) -> list[str]:
        return self.registries.get(registry) or []

    def __contains__(self, registry: str) -> bool:
        return registry in self.registries
