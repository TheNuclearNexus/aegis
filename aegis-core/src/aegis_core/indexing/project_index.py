from heapq import heapify
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import ClassVar

from beet import Context, File, Function, NamespaceFile
from beet.core.utils import extra_field, required_field
from bolt import Module
from mecha import Mecha
from tokenstream import SourceLocation

__all__ = ["FilePointer", "ResourceIndex", "AegisProjectIndex"]

FilePointer = tuple[SourceLocation, SourceLocation]


@dataclass
class ResourceIndice:
    definitions: dict[str, set[FilePointer]] = extra_field(default_factory=dict)
    references: dict[str, set[FilePointer]] = extra_field(default_factory=dict)

    def _dump(self) -> str:
        dump = ""

        dump += "definitions:\n"
        for path, pointers in self.definitions.items():
            for pointer in pointers:
                dump += f"\t- {path} {pointer[0].lineno}:{pointer[0].colno} -> {pointer[1].lineno}:{pointer[1].colno}\n"
        dump += "references:\n"
        for path, pointers in self.references.items():
            for pointer in pointers:
                dump += f"\t- {path} {pointer[0].lineno}:{pointer[0].colno} -> {pointer[1].lineno}:{pointer[1].colno}\n"

        return dump


def valid_resource_location(path: str):
    return bool(re.match(r"^[a-z0-9_\.]+:[a-z0-9_\.]+(\/?[a-z0-9_\.]+)*$", path))


@dataclass
class ResourceIndex:
    _files: dict[str, ResourceIndice] = extra_field(default_factory=dict)
    _lock: Lock = extra_field(default_factory=Lock)

    def remove_associated(self, path: str | File) -> list[str]:
        self._lock.acquire()

        if isinstance(path, File):
            path = str(Path(path.ensure_source_path()).absolute())

        removed = []

        for file, indice in list(self._files.items()):

            if path in indice.references:
                del indice.references[path]
            if path in indice.definitions:
                del indice.definitions[path]

                if len(indice.definitions) == 0:
                    del self._files[file]
                    removed.append(file)

        self._lock.release()
        return removed

    def add_definition(
        self,
        resource_path: str,
        source_path: str,
        source_location: FilePointer = (
            SourceLocation(0, 0, 0),
            SourceLocation(0, 0, 0),
        ),
    ):
        if not valid_resource_location(resource_path):
            raise Exception(f"Invalid resource location {resource_path}")

        self._lock.acquire()

        indice = self._files.setdefault(resource_path, ResourceIndice())
        locations = indice.definitions.setdefault(source_path, set())
        locations.add(source_location)

        self._lock.release()

    def get_definitions(
        self, resource_path: str
    ) -> list[tuple[str, SourceLocation, SourceLocation]]:
        if not (file := self._files.get(resource_path)):
            return []

        definitions = []
        for path, locations in file.definitions.items():
            for location in locations:
                definitions.append((path, *location))

        return definitions

    def get_references(
        self, resource_path: str
    ) -> list[tuple[str, SourceLocation, SourceLocation]]:
        if not (file := self._files.get(resource_path)):
            return []

        references = []
        for path, locations in file.references.items():
            for location in locations:
                references.append((path, *location))

        return references

    def add_reference(
        self,
        resource_path: str,
        source_path: str,
        source_location: FilePointer = (
            SourceLocation(0, 0, 0),
            SourceLocation(0, 0, 0),
        ),
    ):
        if not valid_resource_location(resource_path):
            raise Exception(f"Invalid resource location {resource_path}")

        self._lock.acquire()

        indice = self._files.setdefault(resource_path, ResourceIndice())
        locations = indice.references.setdefault(source_path, set())
        locations.add(source_location)

        self._lock.release()

    def __iter__(self):
        items = self._files.keys()

        for item in items:
            yield item

    def _dump(self) -> str:
        dump = ""

        for file, indice in self._files.items():
            dump += f"\n- '{file}':\n"
            dump += "\t" + "\n\t".join(indice._dump().splitlines())

        return dump


@dataclass
class AegisProjectIndex:
    _ctx: Context
    _resources: dict[type[NamespaceFile], ResourceIndex] = field(default_factory=dict)

    resource_name_to_type: dict[str, type[NamespaceFile]] = field(default_factory=dict)

    def __post_init__(self):
        self.resource_name_to_type = {
            t.snake_name: t for t in self._ctx.get_file_types()
        }

    def __getitem__(self, key: type[NamespaceFile]):
        return self._resources.setdefault(key, ResourceIndex())

    def remove_associated(self, path: str):
        for resource, index in self._resources.items():
            removed_files = index.remove_associated(path)

            for removed in removed_files:
                for pack in self._ctx.packs:
                    if not removed in pack[resource]:
                        continue

                    file = pack[resource][removed]
                    del pack[resource][removed]

                    mecha = self._ctx.inject(Mecha)
                    if file in mecha.database:
                        del mecha.database[file]
                        self._remove_from_queue(file, mecha)

    def _remove_from_queue(self, file, mecha: Mecha):
        index = -1
        for i, (_, _, _, _, queued_file) in enumerate(mecha.database.queue):
            if file == queued_file:
                index = i
                break

        if index != -1:
            mecha.database.queue.pop(i)
            heapify(mecha.database.queue)

    def dump(self) -> str:
        dump = ""
        for resource, index in self._resources.items():
            dump += f"\nResource {resource.__name__}:"
            dump += "\t" + "\n\t".join(index._dump().splitlines())

        return dump
