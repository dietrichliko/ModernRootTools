"""SamplesCache for ModernROOTTools.
"""
import asyncio
import logging
import os
from typing import Any
from typing import Iterator
import fnmatch

import ruamel.yaml as yaml

from mrtools import datasets
from mrtools import exceptions

log = logging.getLogger(__name__)


def _get_dataset_type(data: Any, default: datasets.DatasetType) -> datasets.DatasetType:
    name = data.get("type")
    if name is None:
        return default
    else:
        return datasets.DatasetType.from_string(name)


def _get_str(data: Any, name: str, default: str = "") -> str:
    value = data.get(name)
    return default if value is None else str(value)


class DatasetCache:
    root: datasets.DatasetGroup

    def __init__(self):
        self.root = datasets.DatasetGroup("", datasets.DatasetType.UNKNOWN, None)

    def load(self, dataset_def: str | os.PathLike) -> None:
        """Load datasets from file."""
        log.info("Loading datasets from %s", dataset_def)
        yaml_parser = yaml.YAML(typ="safe")
        with open(dataset_def, "r") as inp:
            for i, data in enumerate(yaml_parser.load_all(inp)):
                try:
                    log.info(
                        "Loading %s for period %s",
                        data.get("name", "undefined"),
                        data.get("period", "undefined"),
                    )
                    self._load_root_dataset(data)
                except exceptions.MRTError as err:
                    log.error("Error %s loading document %d", err, i)
                    pass

        asyncio.run(self.root.afetch_files())

    def _load_root_dataset(self, data: Any) -> datasets.DatasetGroup:
        """Load sample from dict.

        Attributes:
            data: dictionary from yaml

        Raises:
            MRTError on invalid document.
        """
        name = data.get("name")
        if name is None:
            raise exceptions.MRTError("Missing top level name attribute in document.")
        if not self.root.name:
            self.root.name = str(name)
        elif self.root.name != name:
            raise exceptions.MRTError("Inconsistent top level name.")

        period = data.get("period")
        if period is None:
            raise exceptions.MRTError("Missing period attribute.")
        period = str(period)
        if period in self.root._children:
            raise exceptions.MRTError(f"Period {period} already loaded.")

        tree_name = str(data.get("tree_name", "Events"))

        log.debug("Creating DatasetGroup %s/%s", self.root, period)
        sg = datasets.DatasetGroup(
            period, datasets.DatasetType.UNKNOWN, self.root, tree_name
        )

        if "datasets" in data:
            self._load_datasets_groups(data["datasets"], sg)
        if "datasets_from_fs" in data:
            self._load_datasets_from_fs(data["datasets_from_fs"], sg)
        if "datasets_from_eos" in data:
            self._load_datasets_from_eos(data["datasets_from_eos"], sg)
        if "datasets_from_das" in data:
            self._load_datasets_from_das(data["samples_from_das"], sg)

        return sg

    def _load_datasets_groups(self, data: Any, parent: datasets.DatasetGroup) -> None:
        for group_data in data:
            name = _get_str(group_data, "name")
            if not name:
                raise exceptions.MRTError("Dataset group without name")

            log.debug("Creating DatasetGroup %s/%s", parent, name)
            sg = datasets.DatasetGroup(
                name,
                _get_dataset_type(group_data, parent.type),
                parent,
                _get_str(group_data, "tree_name", parent.tree_name),
                _get_str(group_data, "title", name),
                group_data.get("attrs", {}),
            )

            if "datasets" in group_data:
                self._load_datasets_groups(group_data["datasets"], sg)
            if "datasets_from_fs" in group_data:
                self._load_datasets_from_fs(group_data["datasets_from_fs"], sg)
            if "datasets_from_eos" in group_data:
                self._load_datasets_from_eos(group_data["datasets_from_eos"], sg)
            if "datasets_from_das" in group_data:
                self._load_datasets_from_das(group_data["datasets_from_das"], sg)

    def _load_datasets_from_fs(self, data: Any, parent: datasets.DatasetGroup) -> None:
        for d in data:
            name = _get_str(d, "name", "")
            if not name:
                raise exceptions.MRTError("DatasetFromFS without name")

            directory = _get_str(d, "directory", "")
            if not name:
                raise exceptions.MRTError("DatasetFromFS without directory")

            log.debug("Creating DatasetFromFS %s/%s", parent, name)
            datasets.DatasetFromFS(
                name,
                _get_dataset_type(d, parent.type),
                directory,
                _get_str(d, "filter", "*.root"),
                parent,
                _get_str(d, "tree_name", parent.tree_name),
                _get_str(d, "title", name),
                d.get("attrs", {}),
            )

    def _load_datasets_from_eos(self, data: Any, parent: datasets.DatasetGroup) -> None:
        for sample_data in data:
            name = _get_str(sample_data, "name")
            if not name:
                raise exceptions.MRTError("DatasetFromEOS without name")

            directory = _get_str(sample_data, "directory")
            if not name:
                raise exceptions.MRTError(f"DatasetFromEOS {name} without directory")
            if not directory.startswith(("/eos/", "/store/")):
                raise exceptions.MRTError(
                    f"DatasetFromEOS {name} has no valid eos directory {directory}."
                )

            log.debug("Creating DatasetFromEOS %s/%s", parent, name)
            datasets.DatasetFromEOS(
                name,
                _get_dataset_type(sample_data, parent.type),
                directory,
                _get_str(sample_data, "filter", "*.root"),
                parent,
                _get_str(sample_data, "tree_name", parent.tree_name),
                _get_str(sample_data, "title", name),
                sample_data.get("attrs", {}),
            )

    def _load_datasets_from_das(self, data: Any, parent: datasets.DatasetGroup) -> None:
        for sample_data in data:
            name = _get_str(sample_data, "name", "")
            if not name:
                raise exceptions.MRTError("DatasetFromDAS without name")

            dasname = _get_str(sample_data, "dasname", "")
            if not name:
                raise exceptions.MRTError("DatasetFromDAS without dasname")

            if dasname.endswith("/USER"):
                default_instance = "prod/phys03"
            else:
                default_instance = "prod/phys01"

            log.debug("Creating DatasetFromDAS %s/%s", parent, name)
            datasets.DatasetFromDAS(
                name,
                _get_dataset_type(sample_data, parent.type),
                dasname,
                _get_str(sample_data, "instance", default_instance),
                parent,
                _get_str(sample_data, "tree_name", parent.tree_name),
                _get_str(sample_data, "title", name),
                sample_data.get("attrs", {}),
            )

    def verify_period(self, period: list[str]) -> list[str]:
        if period:
            for p in period:
                if p not in self.root._children:
                    raise exceptions.MRTError(f"Unknown period {p}")
            return period
        else:
            return list(self.root._children.keys())

    def list(self, period: str) -> Iterator[datasets.Dataset]:
        try:
            return self.root._children[period].list()
        except KeyError as e:
            raise exceptions.MRTError(f"Unknown period {period}") from e

    def walk(
        self, period: str, topdown: bool = True
    ) -> Iterator[
        tuple[
            datasets.Dataset,
            Iterator[datasets.DatasetGroup],
            Iterator[datasets.DatasetFrom],
        ]
    ]:
        try:
            return self.root._children[period].walk(topdown)
        except KeyError as e:
            raise exceptions.MRTError(f"Unknown period {period}") from e

    def find(self, period: str, pattern: str) -> Iterator[datasets.Dataset]:
        for _, groups, children in self.walk(period):
            for group in groups:
                if fnmatch.fnmatch(group.name, pattern):
                    yield group
            for child in children:
                if fnmatch.fnmatch(child.name, pattern):
                    yield child
