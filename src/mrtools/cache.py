"""Modern ROOT Tools SampleCache."""
import concurrent.futures as futures
import fnmatch
import itertools
import logging
import pathlib
import sys
from collections.abc import Generator
from mrtools import config
from mrtools import model
from types import TracebackType
from typing import Any
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type

import click
import ruamel.yaml as yaml

log = logging.getLogger(__name__)
cfg = config.get()

_click_options: dict[str, Any] = {}

PathOrStr = str | pathlib.Path
PurePathOrStr = str | pathlib.PurePath


class SamplesCache:
    """SampleCache."""

    _root: model.SampleGroup
    _threads: int

    def __init__(self, threads: int = None) -> None:
        """Init samples cache.

        Args:
            threads (int): Threads to query DAS
        """
        self._root = model.SampleGroup("", model.SampleType.UNKNOWN, None)
        if threads is not None:
            self._threads
        elif "cache_threads" in _click_options:
            self._threads = _click_options["cache_threads"]
        else:
            self._threads = cfg.sc.root_threads

    def __enter__(self) -> "SamplesCache":
        """Context manager enter."""
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit."""
        return False

    def get(self, path: PurePathOrStr) -> model.SampleBase | None:
        """Access a sample by its path path."""
        if not isinstance(path, pathlib.PurePath):
            path = pathlib.PurePath(path)

        if path.parts[1] != self._root.name:
            return None

        sample: model.SampleBase | None = self._root
        for name in path.parts[2:]:
            if not isinstance(sample, model.SampleGroup):
                return None
            if (sample := sample.get(name)) is None:
                return None

        return sample

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
        samples: Iterator[model.SampleBase]
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
        pattern: Sequence[str] | str | None = None,
        types: model.SampleTypeSpec | None = None,
    ) -> Generator[model.SampleBase, None, None]:
        """Find samples with a specific name and type in a period.

        Arguments:
            period (str): Running period (empty for all periodes)
            pattern (Sequence[str] | str | None): Sample name (wildcard supported)
            types (model.SampleTypeSpec): Sample type (single value or container)
        """

        def _filter_name(name: str, pattern: Sequence[str] | str | None) -> bool:

            if pattern is None:
                return True
            elif isinstance(pattern, str):
                return fnmatch.fnmatch(name, pattern)
            else:
                return any((fnmatch.fnmatch(name, p) for p in pattern))

        if period:
            root = self._root.get(period)
            if root is None:
                log.error("Period %s not found", period)
                return
        else:
            root = self._root

        for _, samples, groups in model.walk(root):
            for s in itertools.chain(samples, groups):
                if _filter_name(s.name, pattern) and model.filter_types(s, types):
                    yield s

    def walk(
        self, period: str = "", topdown=True
    ) -> Generator[
        Tuple[model.SampleBase, Iterator[model.Sample], Iterator[model.SampleGroup]],
        None,
        None,
    ]:
        """Walk samples tree."""
        if period:
            root = self._root.get(period)
            if root is None:
                log.error("Period %s not found", period)
                return
        else:
            root = self._root

        for y in model.walk(root, topdown):
            yield y

    def print_tree(
        self,
        node: model.SampleBase = None,
        indent: int = 0,
        file: TextIO = sys.stdout,
    ) -> None:
        """Print samples tree."""
        if node is None:
            node = self._root

        print(f"{' '*indent}{node} ({len(node)})", file=file)

        for sample in node.children:
            self.print_tree(sample, indent + 2, file=file)


def click_options():
    """Clock options for cache."""

    def _set_threads(ctx, param, value):
        _click_options["cache_threads"] = int(value)

    def decorator(f):

        return click.option(
            "--cache-threads",
            metavar="THREADS",
            callback=_set_threads,
            expose_value=False,
            default=4,
            type=click.IntRange(1),
            help="Cache threads",
        )(f)

    return decorator
