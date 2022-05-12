"""Modern ROOT Tools SampleCache."""
import concurrent.futures as futures
import fnmatch
import itertools
import logging
import os
import pathlib
import sys
from collections.abc import Container
from collections.abc import Generator
from contextlib import AbstractContextManager
from mrtools import configuration
from mrtools import model
from types import TracebackType
from typing import Dict
from typing import Literal
from typing import Optional
from typing import TextIO
from typing import Type
from typing import Union

# For CMSSW ruamel.yaml has been renamed yaml ....
if "CMSSW_BASE" in os.environ:
    import yaml  # type: ignore
else:
    import ruamel.yaml as yaml

log = logging.getLogger(__name__)
config = configuration.get()

PathOrStr = Union[str, pathlib.Path]


class SamplesCache(AbstractContextManager):
    """SampleCache."""

    _root: model.SampleGroup
    _threads: int

    def __init__(self, threads: int = 4) -> None:
        self._root = model.SampleGroup("", model.SampleType.UNKNOWN, None)
        self._threads = threads or config.sc.threads

    def __enter__(self) -> "SamplesCache":
        """Context manager enter."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Literal[False]:
        """Context manager exit."""
        return False

    def load(self, input: PathOrStr) -> None:
        """Load and prepare samples from YAML file.

        The yaml file can congtain several documents representing various periods.

        Arguments:
            input: Yaml file.
        """
        self._yaml_load(input)

        with futures.ThreadPoolExecutor(max_workers=self._threads) as e:
            tasks: Dict[futures.Future, model.Sample] = {}
            for _, samples, _ in model.walk(self._root):
                tasks |= {e.submit(lambda x: x.get_files(), s): s for s in samples}

            for f in futures.as_completed(tasks.keys()):
                sample = tasks[f]
                try:
                    f.result()
                except FileNotFoundError:
                    pass
                if len(sample):
                    log.debug("Sample %s has %d files.", sample, len(sample))
                else:
                    if isinstance(sample, model.SampleFromFS):
                        log.warning(
                            "Sample %s is empty. Check directory %s,",
                            sample,
                            sample._directory,
                        )
                    elif isinstance(sample, model.SampleFromDAS):
                        log.warning(
                            "Sample %s is empty. Check DASname %s.",
                            sample,
                            sample._dasname,
                        )
                    else:
                        log.warning("Sample %s is empty.", sample)

    def _yaml_load(self, input: PathOrStr) -> None:
        """Load samples from YAML file.

        The yaml file can contain several documents representing various periods.

        Arguments:
            input: Yaml file.
        """
        yaml_parser = yaml.YAML(typ="safe")
        with open(input, "r") as inp:
            for i, data in enumerate(yaml_parser.load_all(inp)):
                log.info("Loading yaml document %d from %s", i, input)

                name = data.get("name")
                if name is None:
                    log.error("Missing top level name attribute.")
                    continue
                if not self._root._name:
                    self._root._name = name
                elif self._root.name != name:
                    log.error("Wrong top level name")
                    continue

                period = data.get("period")
                if period is None:
                    log.error("Missing period attribute.")
                    continue
                if self._root.get(period) is not None:
                    log.error("Period %s already loaded.", period)

                group = model.SampleGroup(period, model.SampleType.UNKNOWN, self._root)
                group.append(data.get("samples"))

    def list(
        self,
        period: str = "",
        types: model.SampleTypeSpec = None,
    ) -> Generator[model.SampleBase, None, None]:
        """List top level samples.

        Arguments:
            period: datataking periode (none for all periods)
            types: Sample type (single value or container)
        """
        if not period:
            samples = itertools.chain.from_iterable(
                (s.children for s in self._root.children)
            )
        else:
            root = self._root.get(period)
            if root is None:
                log.error("Period %s not found", period)
                return
            samples = root.children

        yield from (s for s in samples if model.filter_types(s, types))


    def _filter_name(name: str, pattern: Union[str, List[str]])-> bool:

        if isinstance(pattern, list):
            return any(( fnmatch.fnmatch(name, p) for p in pattern))
        else:
            return fnmatch.fnmatch(name, pattern)
    def find(
        self,
        period: str = "",
        name: Union[str, List[str]] = "*",
        types: model.SampleTypeSpec = None,
    ) -> Generator[model.SampleBase, None, None]:
        """Find samples with a specific name and type in a period.

        Arguments:
            period: Running period (empty for all periodes)
            name: Sample name (wildcard supported)
            types: Sample type (single value or container)
        """
        root = self._root.get(period) if period else self._root
        if root is None:
            log.error("Period %s not found", period)
            return

        for _, samples, groups in model.walk(root):
            for s in itertools.chain(samples, groups):
                if fnmatch.fnmatch(s.name, name) and model.filter_types(s, types):
                    yield s

    def print_tree(
        self,
        node: Optional[model.SampleBase] = None,
        indent: int = 0,
        file: Optional[TextIO] = sys.stdout,
    ) -> None:
        """Print samples tree."""

        if node is None:
            node = self._root

        print(f"{' '*indent}{node} ({len(node)})", file=file)

        for sample in node.children:
            self.print_tree(sample, indent + 2, file=file)
