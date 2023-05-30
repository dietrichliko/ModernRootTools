"""Modern ROOT Tools Samples.

Datasets are collection of files associated to a
run period or a MC production.

Based on DatasetGroups trees of datasets can be
constructed, combining relates datasets.

Class Hierarchy

    Dataset
    DatasetGroup
    DatasetBase
        DatasetFromFS
        DatasetFromEOS
           DatasetFromDAS
    SubDataset

"""
import abc
import asyncio
import enum
import logging
import os
import sys
import json
from typing import Any
from typing import Iterator
from typing import cast
from typing import Optional
from typing import Container
import pathlib
import itertools
import fnmatch

from mrtools import utils
from mrtools import config

import ROOT

BLOCKSIZE = 256 * 1024 * 1024

log = logging.getLogger(__name__)
cfg = config.get()


class FileInDataset:
    """File as part of a dataset.

    Attributes:
        dataset: Dataset
        dirname: Directory Name
        name: File Name
        size: Filesize
        entries: TTree entries
        checksum: Adler32
    """

    dataset: "DatasetFrom"
    dirname: str
    name: str
    size: int
    entries: int | None
    checksum: int | None

    xrdcp_sem = asyncio.Semaphore(cfg.max_xrdcp)

    def __init__(
        self,
        dataset: "DatasetFrom",
        dirname: str,
        name: str,
        size: int,
        *,
        entries: int | None = None,
        checksum: int | None = None,
    ) -> None:
        """Init."""
        self.dataset = dataset
        self.dirname = sys.intern(dirname)
        self.name = name
        self.size = size
        self.entries = entries
        self.checksum = checksum

    def __str__(self) -> str:
        return f"{self.dirname}/{self.name}"

    def __repr__(self) -> str:
        r = [
            f"path={self}",
            f"size={utils.human_readable_size(self.size)}",
        ]
        if self.entries:
            r.append(f"entries={self.entries}")
        if self.checksum:
            r.append(f"checksum={self.checksum:08X}")

        return f"FileInDataset({', '.join(r)})"

    def path(self, stage: bool | None = None) -> pathlib.Path:
        """Path on Filesystem."""
        return self.dataset._resolve_path(self, stage)

    def url(self, stage: bool | None = None) -> str:
        """Access url."""
        return self.dataset._resolve_url(self, stage)

    def get_entries(self) -> int:
        """Get TTree entries."""
        tree = ROOT.TTree(self.dataset.tree_name, self.dataset._resolve_url(self))
        return cast(int, tree.GetEntries())

    async def astage(self) -> None:
        """Stage a file."""
        async with FileInDataset.xrdcp_sem:
            log.debug("Stageing %s ...", self)
            cmd = [
                "--nopbar",
                "--force",
                "--retry",
                str(cfg.xrdcp_retry),
            ]
            if self.checksum is not None:
                cmd += ["--cksum", f"adler32:{self.checksum:08x}"]
            cmd += [self.url(stage=False), str(self.path(stage=True))]

            proc = await asyncio.create_subprocess_exec(cfg.xrdcp, *cmd)
            await proc.wait()

            if proc.returncode != 0:
                raise RuntimeError("xrdcp failed %d", proc.returncode)

    async def acleanup(self) -> None:
        log.debug("Cleanup %s ...", self)
        path = self.path(stage=True)
        if path.exists():
            os.unlink(path)


class DatasetType(enum.Enum):
    """Dataset types."""

    UNKNOWN = 0
    DATA = 1
    SIMULATION = 2
    BACKGROUND = 3
    SIGNAL = 4

    def is_of(self, sample_type: "DatasetTypeSpec") -> bool:
        return (
            sample_type is None
            or isinstance(sample_type, Container)
            and self in sample_type
            or self == sample_type
        )

    @staticmethod
    def from_string(name: str) -> "DatasetType":
        name = name.upper()
        for st in DatasetType:
            if st.name == name:
                return st
        raise ValueError(f"{name} is not a valid SampleType")


DatasetTypeSpec = DatasetType | Container[DatasetType] | None


class Dataset:
    """Base class for all samples.

    Attributes:
        name: Sample name
        type: Data, Simulation, ....
        parent: Parent sample group
        tree_name: ROOT TTree name
        attrs: Dictionary of arbitrary attributes
    """

    name: str
    type: DatasetType
    parent: Optional["DatasetGroup"]
    tree_name: str
    title: str
    attrs: dict

    def __init__(
        self,
        name: str,
        type: DatasetType,
        parent: Optional["DatasetGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Base class for all Datasets

        Args:
            name: Dataset name
            type: Data, Simulation, Background, Signal ...
            parent: Parent dataset group
            tree_name: Name of ROOT TTree
            title: Human readable name for sample for plots
            attrs: A dictionary of arbitrary attributes
        """
        self.name = name
        self.type = type
        self.parent = parent
        if parent is not None:
            parent._children[name] = self
        self.tree_name = tree_name
        self.title = title
        self.attrs = {} if attrs is None else attrs

    @property
    def parts(self) -> list[str]:
        parts = [self.name]
        p = self.parent
        while p is not None:
            parts.insert(0, p.name)
            p = p.parent
        return parts

    def __str__(self) -> str:
        return os.path.join("/", *self.parts)

    @property
    def path(self) -> pathlib.PurePath:
        """Sample path as Path."""
        return pathlib.Path(*self.parts)

    @property
    def period(self) -> str:
        return self.parts[2]

    @abc.abstractmethod
    def list(self) -> Iterator["Dataset"]:
        """Iterator on children samples."""
        pass

    @abc.abstractmethod
    def walk(
        self, topdown: bool = True
    ) -> Iterator[tuple["Dataset", Iterator["DatasetGroup"], Iterator["DatasetFrom"]]]:
        """Iterator on children samples."""
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator["FileInDataset"]:
        """Iterator on contained files."""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number on contained files."""
        pass

    @abc.abstractmethod
    def __contains__(self, item: Any) -> bool:
        """Item is file of the sample."""
        pass

    @property
    def entries(self) -> int | None:
        """Number of entries of the trees."""
        try:
            return sum(f.entries for f in self)  # type: ignore
        except TypeError:
            return None

    @property
    def size(self) -> int:
        """Size of the contained files."""
        return sum(f.size for f in self)

    def chain(self, max_files: int = 0, stage: bool | None = None) -> Any:
        """Iterator on url of the contained files."""

        chain = ROOT.TChain(self.tree_name)
        file_iter = iter(self) if max_files <= 0 else itertools.islice(self, max_files)
        for f in file_iter:
            chain.Add(f.url(stage))
        return chain

    @abc.abstractmethod
    async def afetch_files(self) -> None:
        pass

    @abc.abstractmethod
    async def astage(self) -> None:
        pass

    @abc.abstractmethod
    async def acleanup(self) -> None:
        pass


class DatasetFrom(Dataset):
    """Base class for individual samples.

    Attributes:
       _files: two level dictionary of files
    """

    _files: list["FileInDataset"]

    def __init__(
        self,
        name: str,
        type: DatasetType,
        parent: Optional["DatasetGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, type, parent, tree_name, title, attrs)
        self._files = []

    def __iter__(self) -> Iterator["FileInDataset"]:
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str | os.PathLike):
            head, tail = os.path.split(item)
            return any(f.dirname == head and f.name == tail for f in self._files)
        return False

    def list(self) -> Iterator["Dataset"]:
        return
        yield

    def walk(
        self, topdown: bool = True
    ) -> Iterator[tuple["Dataset", Iterator["DatasetGroup"], Iterator["DatasetFrom"]]]:
        """Iterator on children samples."""
        return
        yield

    def get(self, name: str) -> Dataset | None:
        return None

    @abc.abstractmethod
    def _resolve_path(
        self, file: "FileInDataset", stage: bool | None = None
    ) -> pathlib.Path:
        pass

    @abc.abstractmethod
    def _resolve_url(self, file: "FileInDataset", stage: bool | None = None) -> str:
        pass


class DatasetFromFS(DatasetFrom):
    """
    Dataset defined by files in directories in the file system.

    its content is defined by the files contained in directory
    and all subdirectories, which pass the filename filter (*.root)

    Attributes:
        directory: the directory containing the dataset
        filter: filename filter (default *.root)
    """

    directory: str
    filter: str

    def __init__(
        self,
        name: str,
        type: DatasetType,
        directory: str,
        filter: str = "*.root",
        parent: Optional["DatasetGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, type, parent, tree_name, title, attrs)
        self.directory = directory
        self.filter = filter

    def _resolve_path(
        self, file: FileInDataset, stage: bool | None = None
    ) -> pathlib.Path:
        return pathlib.Path(file.dirname, file.name)

    def _resolve_url(self, file: FileInDataset, stage: bool | None = None) -> str:
        return f"file://{file.dirname}/{file.name}"

    async def afetch_files(self) -> None:
        self._files = []

        for root, _dirs, files, root_fd in os.fwalk(self.directory):
            for name in files:
                if fnmatch.fnmatch(name, self.filter):
                    size = os.stat(name, dir_fd=root_fd).st_size
                    self._files.append(FileInDataset(self, root, name, size))

    async def astage(self) -> None:
        pass

    async def acleanup(self) -> None:
        pass


class DatasetFromEOS(DatasetFrom):
    directory: str
    filter: str

    def __init__(
        self,
        name: str,
        type: DatasetType,
        directory: str,
        filter: str = "*.root",
        parent: Optional["DatasetGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, type, parent, tree_name, title, attrs)
        self.directory = directory
        self.filter = filter

    def _resolve_path(
        self, file: FileInDataset, stage: bool | None = None
    ) -> pathlib.Path:
        stage_path = pathlib.Path(cfg.cache_path, file.dirname[1:], file.name)
        if stage or stage is None and stage_path.exists():
            return stage_path
        else:
            if file.dirname.startswith("/store/"):
                return pathlib.Path(cfg.store_path, file.dirname[1:], file.name)
            else:
                return pathlib.Path(file.dirname, file.name)

    def _resolve_url(self, file: FileInDataset, stage: bool | None = None) -> str:
        stage_path = pathlib.Path(cfg.cache_path, file.dirname[1:], file.name)
        if stage or stage is None and stage_path.exists():
            return f"file://{stage_path}"
        else:
            if file.dirname.startswith("/store/"):
                path = pathlib.Path(cfg.store_path, file.dirname[1:], file.name)
                if path.exists():
                    return f"{cfg.local_url}/{path}"
                else:
                    return f"{cfg.global_url}/{path}"
            else:
                return f"{cfg.local_url}/{file.dirname}/{file.name}"

    async def afetch_files(self) -> None:
        self._files = []

        if self.directory.startswith("/store/"):
            path = os.path.join(cfg.store_path, self.directory[1:])
        else:
            path = self.directory

        for root, _dirs, files, root_fd in os.fwalk(path):
            for name in files:
                if fnmatch.fnmatch(name, self.filter):
                    size = os.stat(name, dir_fd=root_fd).st_size
                    checksum = int(
                        os.getxattr(os.path.join(path, name), "eos.checksum").decode(
                            "utf-8"
                        ),
                        16,
                    )
                    self._files.append(
                        FileInDataset(self, root, name, size, checksum=checksum)
                    )

    async def astage(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for file in self:
                tg.create_task(file.astage())

    async def acleanup(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for file in self:
                tg.create_task(file.acleanup())


class DatasetFromDAS(DatasetFrom):
    dasname: str
    instance: str

    dasgoclient_sem = asyncio.Semaphore(cfg.max_dasgoclient)

    def __init__(
        self,
        name: str,
        type: DatasetType,
        dasname: str,
        instance: str,
        parent: Optional["DatasetGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, type, parent, tree_name, title, attrs)
        self.dasname = dasname
        self.filter = instance

    def _resolve_path(
        self, file: FileInDataset, stage: bool | None = None
    ) -> pathlib.Path:
        stage_path = pathlib.Path(cfg.cache_path, file.dirname[1:], file.name)
        if stage or stage is None and stage_path.exists():
            return stage_path
        else:
            if file.dirname.startswith("/store/"):
                return pathlib.Path(cfg.store_path, file.dirname[1:], file.name)
            else:
                return pathlib.Path(file.dirname, file.name)

    def _resolve_url(self, file: FileInDataset, stage: bool | None = None) -> str:
        stage_path = pathlib.Path(cfg.cache_path, file.dirname[1:], file.name)
        if stage or stage is None and stage_path.exists():
            return f"file://{stage_path}"
        else:
            if file.dirname.startswith("/store/"):
                path = pathlib.Path(cfg.store_path, file.dirname[1:], file.name)
                if path.exists():
                    return f"{cfg.local_url}/{path}"
                else:
                    return f"{cfg.global_url}/{path}"
            else:
                return f"{cfg.local_url}/{file.dirname}/{file.name}"

    async def afetch_files(self) -> None:
        self._files = []
        async with DatasetFromDAS.dasgoclient_sem:
            proc = await asyncio.create_subprocess_exec(
                cfg.dasgoclient,
                "--json",
                f"--query=file dataset={self.dasname} instance={self.instance}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            for item in json.loads(stdout):
                for file_item in item["file"]:
                    root, name = os.path.split(file_item["name"])
                    self._files.append(
                        FileInDataset(
                            self,
                            root,
                            name,
                            int(file_item["size"]),
                            entries=int(file_item["nevents"]),
                            checksum=int(file_item["adler32"], 16),
                        )
                    )

    async def astage(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for file in self:
                tg.create_task(file.astage())

    async def acleanup(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for file in self:
                tg.create_task(file.acleanup())


class DatasetGroup(Dataset):
    _children: dict[str, Dataset]

    def __init__(
        self,
        name: str,
        type: DatasetType,
        parent: Optional["DatasetGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Initialise a dataset group.

        Arguments:
            name: Dataset name.
            type: Type of the dataset (Data/Simulation/Background/Signal).
            parent: Parent dataset group.
            tree_name: Name of the ROOT TTree
            title: Name of the dataset in ROOT latex format.
            attrs: User defined attributes.
        """
        super().__init__(name, type, parent, tree_name, title, attrs)
        self._children = {}

    def list(self) -> Iterator["Dataset"]:
        return iter(self._children.values())

    def walk(
        self, topdown: bool = True
    ) -> Iterator[tuple["Dataset", Iterator["DatasetGroup"], Iterator["DatasetFrom"]]]:
        """Iterator on child datasets."""
        if topdown:
            yield (
                self,
                (s for s in self._children.values() if isinstance(s, DatasetGroup)),
                (s for s in self._children.values() if isinstance(s, DatasetFrom)),
            )
        for child in self._children.values():
            for x in child.walk(topdown):
                yield x
        if not topdown:
            yield (
                self,
                (s for s in self._children.values() if isinstance(s, DatasetGroup)),
                (s for s in self._children.values() if isinstance(s, DatasetFrom)),
            )

    def __iter__(self) -> Iterator["FileInDataset"]:
        """Iterator on contained files."""
        return itertools.chain.from_iterable(s for s in self._children.values())

    def __len__(self) -> int:
        """Number on contained files."""
        return sum(len(s) for s in self._children.values())

    def __contains__(self, item: Any) -> bool:
        """Item is file of the dataset."""
        return any(item in s for s in self._children.values())

    async def afetch_files(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for child in self._children.values():
                tg.create_task(child.afetch_files())

    async def astage(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for child in self._children.values():
                tg.create_task(child.astage())

    async def acleanup(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for child in self._children.values():
                tg.create_task(child.acleanup())
