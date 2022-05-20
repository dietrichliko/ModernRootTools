"""Modern ROOT Tools SampleCache."""
import concurrent.futures as futures
import fnmatch
import itertools
import logging
import os
import pathlib
import sys
from collections.abc import Generator
from contextlib import AbstractContextManager
from mrtools import configuration
from mrtools import model
from types import TracebackType
from typing import List
from typing import Literal
from typing import Optional
from typing import TextIO
from typing import Type
from typing import Union

# For CMSSW ruamel.yaml has been renamed yaml ....
if "CMSSW_BASE" in os.environ:
    import yaml  # type: ignore
else:
    import ruamel.yaml as yaml  # type: ignore

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
            tasks: dict[futures.Future, model.Sample] = {}
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
                    log.error("Sample %s is empty.", sample)
                    if isinstance(sample, model.SampleFromFS):
                        log.error("Check directory %s.", sample._directory)
                    elif isinstance(sample, model.SampleFromDAS):
                        log.error("Check DASname %s.", sample._dasname)
                    else:
                        log.warning("Sample %s is empty.", sample)

    def _yaml_load(self, input: PathOrStr) -> None:
        """Load samples from YAML file.

        The yaml file can contain several documents representing various periods.

        Arguments:
            input: Yaml file.
        """
        log.info("Loading samples from %s", input)
        yaml_parser = yaml.YAML(typ="safe")
        with open(input, "r") as inp:
            for data in yaml_parser.load_all(inp):

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

                log.info("Loading %s for %s", name, period)

                group = model.SampleGroup(period, model.SampleType.UNKNOWN, self._root)
                group.load(data.get("samples"))

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

    def find(
        self,
        period: str = "",
        pattern: Union[None, str, List[str]] = None,
        types: model.SampleTypeSpec = None,
    ) -> Generator[model.SampleBase, None, None]:
        """Find samples with a specific name and type in a period.

        Arguments:
            period: Running period (empty for all periodes)
            pattern: Sample name (wildcard supported)
            types: Sample type (single value or container)
        """
        def _filter_name(name: str, pattern: Union[None, str, List[str]])-> bool:

            if pattern is None:
                return True
            elif isinstance(pattern, list):
                return any(( fnmatch.fnmatch(name, p) for p in pattern))
            else:
                return fnmatch.fnmatch(name, pattern)
                
        root = self._root.get(period) if period else self._root
        if root is None:
            log.error("Period %s not found", period)
            return

        for _, samples, groups in model.walk(root):
            for s in itertools.chain(samples, groups):
                if _filter_name(s.name, pattern) and model.filter_types(s, types):
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
