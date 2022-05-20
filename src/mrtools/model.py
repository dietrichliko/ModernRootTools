"""Modern ROOT Tools Sample definitions.

Class Hierarchy:

- SampleBase
    - SampleGroup
    - Sample
       - SampleFromFS
       - SampleFromDAS

Sample Tree:

root (Sample Group)

   - period A (Sample Group)

     - data (Sample Group)
       - data sample A (SampleFromFS/SampleFromDAS)
       - data sample B
       - ...

     - background A (SampleGroup)
       - mc sample A (SampleFromFS/SampleFromDAS)
       - mc sample B (SampleFromFS/SampleFromDAS)

   - period A (Sample Group)
     - ...
"""
import abc
import enum
import fnmatch
import itertools
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Generator
from mrtools import configuration
from mrtools import exceptions
from mrtools import utilities
from typing import Any, Sequence
from typing import cast
from typing import Container
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Union

import ROOT
from datasize import DataSize

log = logging.getLogger(__name__)
config = configuration.get()

PathOrStr = Union[str, pathlib.Path]
PurePathOrStr = Union[str, pathlib.PurePath]

SampleTypeSpec = Union[None, "SampleType", Container["SampleType"]]


class SampleFileState(enum.Enum):
    """File state."""

    OK = 0
    STAGING = 1
    STAGED = 2
    ERROR = 3


class SampleFile:
    """File as part of sample."""

    _sample: "SampleBase"
    _dirname: str
    _name: str
    _size: DataSize
    _state: SampleFileState
    _entries: Optional[int]
    _checksum: Optional[int]
    _attrs: dict[str, Any]
    _price: Optional[float]
    _value: Optional[float]
    _ts: Optional[float]

    def __init__(
        self,
        sample: "SampleBase",
        path: PurePathOrStr,
        size: Union[int, DataSize],
        *,
        state: SampleFileState = SampleFileState.OK,
        entries: Optional[int] = None,
        checksum: Optional[int] = None,
        price: Optional[float] = None,
        value: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Init File.

        Arguments:
            sample: Sample to which the file is associated.
            path: File path (absolute path or CMS /store/..)
            size: File size,
            state: File state (ok/stageing/staged/error).
            entries: Number of entries of the contained tree.
            checksum: Adler32 checksum of file.
            price: The price of a file for staging.
            value: The reduced value of a file due to ageing.
            ts: Timestamp of the last update of the value.
        """
        self._sample = sample
        dirname, name = os.path.split(path)
        self._dirname = sys.intern(dirname)
        self._name = name
        self._size = size if isinstance(size, DataSize) else DataSize(size)
        self._state = state
        self._entries = entries
        self._checksum = checksum
        self._price = price
        self._value = value
        self._ts = ts

    def __str__(self) -> str:
        return os.path.join(self._dirname, self._name)

    def __repr__(self) -> str:
        r = [
            f'path="{self}"',
            f"size={self.size:.2a}",
            f"state={self._state.name}",
        ]
        if self._entries:
            r.append(f"entries={self._entries}")
        if self._checksum:
            r.append(f"checksum={self._checksum:08X}")
        if self._price:
            r.append(f"price={self._price!r}")
        if self._value:
            r.append(f"value={self._value!r}")
        if self._ts:
            r.append(f"tsp={self._ts!r}")

        return f"SampleFile({', '.join(r)})"

    @property
    def size(self) -> DataSize:
        """Size of file in bytes."""
        return self._size

    @property
    def entries(self) -> Optional[int]:
        """Number of entries of the ROOT tree."""
        return self._entries

    @property
    def checksum(self) -> Optional[int]:
        """Adler32 checksum of the file."""
        return self._checksum

    @property
    def url(self) -> str:
        """The file url.

        Returns:
            Url to access the file with ROOT.

        Raises:
            MRTError for file in state STAGING or ERROR
        """
        if self._state == SampleFileState.OK:
            if self._dirname.startswith("/store/"):
                path = f"{config.site.store_path}{self._dirname}/{self._name}"
                if os.path.exists(path):
                    return f"{config.site.local_url}/{path}"
                else:
                    return f"{config.site.global_url}/{path}"
            elif self._dirname.startswith("/eos/"):
                path = f"{self._dirname}/{self._name}"
                if os.path.exists(path):
                    return f"{config.site.local_url}/{path}"
                else:
                    return f"{config.site.global_url}/{path}"
            else:
                return f"file://{self._dirname}/{self._name}"
        elif self._state == SampleFileState.STAGED:
            return f"file://{config.site.cache_path}{self._dirname[1:]}/{self._name}"
        else:
            raise exceptions.MRTError("File %s is in state %s", self, self._state.name)

    def get_entries(self) -> int:
        """Determine the actual numbers of entries from the file."""
        f = ROOT.TFile(self.url, "READ")
        tree = f.Get(self._sample.tree_name)
        entries = cast(int, tree.GetEntries())
        f.Close()

        log.debug("File %s has %d entries", str(self), entries)

        return entries

    def get_size(self) -> DataSize:
        """Get the actual file size."""
        if self._dirname.startswith("/store/"):
            path = f"{config.site.store_path}{self._dirname}/{self._name}"
        else:
            path = f"{self._dirname}/{self._name}"
        if not os.path.exists(path):
            raise exceptions.MRTError(
                f"File {self} is not local. Cannot determine size."
            )
        size = DataSize(os.stat(path).st_size)

        log.debug("File %s has %s", self, format(size, "a"))

        return size

    def get_checksum(self, from_data: bool = False) -> int:
        """Determine checksum from EOS metadata.

        Arguments:
            from_data: Recalculate checksum from data.

        Returns:
            Adler32 checksum.

        Raises:
            MRTError: failed to retrieve checksum
        """
        if from_data:
            return utilities.xrd_checksum(self.url)
        else:
            try:
                checksum = int(
                    os.getxattr(str(self), "eos.checksum").decode("utf-8"), 16
                )
            except OSError as exc:
                raise exceptions.ModelError(
                    f"Could not retrieve the checksum metadata for {self}"
                ) from exc

        log.debug("File %s has checksum %x", str(self), checksum)

        return checksum

    def stage(self) -> None:
        """Stage the file."""
        if self._state != SampleFileState.OK:
            raise exceptions.ModelError(
                "File {self} in state {self._state.name} and cannot be staged."
            )
        stage_path = os.path.join(config.site.cache_path, str(self)[1:])
        log.debug("Stageing %s ...", self)
        cmd = [
            config.bin.xrdcp,
            "--nopbar",
            "--force",
            "--retry",
            str(config.sc.xrdcp_retry),
        ]
        if self.checksum is not None:
            cmd += ["--cksum", f"adler32:{self.checksum:08x}"]
        cmd += [self.url, str(stage_path)]
        start_time = time.time()
        subprocess.check_call(cmd)  # noqa:S603
        total_time = time.time() - start_time
        rate = f"{DataSize(self.size/total_time):.2A}"
        log.debug("File %s is staged (%sB/sec)", self, rate)

        self._state = SampleFileState.STAGED
        self._price = total_time
        self._value = self._price
        self._ts = time.time()


class SampleType(enum.Enum):
    """Sample types."""

    UNKNOWN = 0
    DATA = 1
    SIMULATION = 2
    BACKGROUND = 3
    SIGNAL = 4


class SampleBase(Collection):
    """Base class for all sample types."""

    _name: str
    _type: SampleType
    _parent: Optional["SampleBase"]
    _tree_name: str
    _title: str
    _attrs: dict[str, Any]

    def __init__(
        self,
        name: str,
        type: SampleType,
        parent: Optional["SampleGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialise the base class.

        Arguments:
            name: Sample name.
            type: Type of the sample(Data/Simulation/Background/Signal).
            parent: Parent Samplegroup.
            tree_name: Name of the ROOT TTree
            title: Name of the sample in ROOT latex format.
            attrs: User defined attributes.
        """
        self._name = name
        self._type = type
        self._parent = parent
        if parent is not None:
            parent._children[name] = self
        self._tree_name = tree_name or "Events"
        self._title = title or name
        self._attrs = attrs or {}

    def __str__(self) -> str:
        """Path of the sample as str."""
        path = [self._name]
        p = self._parent
        while p is not None:
            path.insert(0, p.name)
            p = p.parent
        path.insert(0, "")
        return "/".join(path)

    @property
    def name(self) -> str:
        """Name of the sample."""
        return self._name

    @property
    def title(self) -> str:
        """Nice name of the sample."""
        return self._title

    @property
    def path(self) -> pathlib.PurePath:
        """Path of the sample."""
        return pathlib.PurePath(str(self))

    @property
    def parent(self) -> Optional["SampleBase"]:
        """Parent of the sample."""
        return self._parent

    @property
    @abc.abstractmethod
    def children(self) -> Iterator["SampleBase"]:
        """Iterator on children samples."""
        pass

    @property
    def type(self) -> SampleType:
        """Sample type."""
        return self._type

    @property
    def tree_name(self) -> str:
        """Name of ROOT TTree."""
        return self._tree_name

    @property
    def attrs(self) -> dict[str, Any]:
        """User defined attributes."""
        return self._attrs

    @abc.abstractmethod
    def __iter__(self) -> Iterator["SampleFile"]:
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
    def entries(self) -> Optional[int]:
        """Number of entries of the trees."""
        try:
            return sum(f._entries for f in iter(self))
        except TypeError:
            return None

    @property
    def size(self) -> DataSize:
        """Size of the contained files."""
        return DataSize(sum(f._size for f in iter(self)))

    def chain(self, max: Optional[int] = None) -> Any:
        """Iterator on url of the contained files."""
        url_iter: Iterator[str] = (f.url for f in self)
        if max:
            url_iter = itertools.islice(url_iter, max)

        chain = ROOT.TChain(self.tree_name)
        for url in url_iter:
            chain.Add(url)

        return chain


class Sample(SampleBase):
    """Common base class for real samples."""

    _files: dict[str, dict[str, SampleFile]]

    def __init__(
        self,
        name: str,
        type: SampleType,
        parent: "SampleGroup",
        tree_name: str = "",
        title: str = "",
        attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialise a sample base class.

        Arguments:
            name: Sample name.
            type: Type of the sample(Data/Simulation/Background/Signal).
            parent: Parent Samplegroup.
            tree_name: Name of the ROOT TTree
            title: Name of the sample in ROOT latex format.
            attrs: user defined attributes.
        """
        super().__init__(name, type, parent, tree_name, title, attrs)

        self._files = defaultdict(dict)

    def __iter__(self) -> Iterator[SampleFile]:
        """Iterator on contained files."""
        return itertools.chain.from_iterable(f.values() for f in self._files.values())

    def __len__(self) -> int:
        """Number of contained files."""
        return sum(len(f) for f in self._files.values())

    def __contains__(self, item: Any) -> bool:
        """Path item contained in sample."""
        dirname, name = os.path.split(item)
        if dirname in self._files:
            return name in self._files[dirname]
        else:
            return False

    @property
    def children(self) -> Iterator["SampleBase"]:
        """Iterator on children samples."""
        return iter([])

    def insert(self, file: SampleFile) -> None:
        """Insert a file."""
        self._files[file._dirname][file._name] = file

    @abc.abstractmethod
    def get_files(self) -> None:
        """Get files from the definition."""
        pass


class SampleFromFS(Sample):
    """Sample given by directory."""

    _directory: pathlib.Path
    _filter: str

    def __init__(
        self,
        name: str,
        type: SampleType,
        parent: "SampleGroup",
        directory: PathOrStr,
        filter: str = "",
        tree_name: str = "",
        title: str = "",
        attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialise a sample base class.

        Arguments:
            name: Sample name.
            type: Type of the sample(Data/Simulation/Background/Signal).
            parent: Parent Samplegroup.
            directory: Directory where files are stored.
            filter: Filename filter (default *.root).
            tree_name: Name of the ROOT TTree.
            title: Name of the sample in ROOT latex format.
            attrs: User defined attributes.
        """
        super().__init__(name, type, parent, tree_name, title, attrs)

        self._directory = pathlib.Path(directory)
        self._filter = filter or "*.root"

    def __repr__(self) -> str:

        items = [
            f"path={self.path}",
            f"type={self._type.name}",
            f"directory={self._directory}",
            f"filter={self._filter}",
            f"#files={len(self)}",
            f"size={self.size:.2a}",
        ]
        if entries := self.entries:
            items.append(f"entries={entries}")
        for key, value in self.attrs.items():
            items.append(f"{key}={value}")
        return f"SampleFromFS({', '.join(items)})"

    def get_files(self) -> None:
        """List the files in directory and create file structure."""
        # HACK
        dirpath = pathlib.Path(
            "/scratch-cbe/users/dietrich.liko", *self._directory.parts[4:]
        )
        # if self._directory.parts[1] == "store":
        #     dirpath = config.site.store_path / self._directory.relative_to("/")
        # else:
        #     dirpath = self._directory

        for root, _dirs, files, rootfd in os.fwalk(dirpath):
            for filename in files:
                if not fnmatch.fnmatch(filename, self._filter):
                    continue
                path = pathlib.Path(root, filename)
                size = os.stat(filename, dir_fd=rootfd).st_size
                if path.parts[1] == "eos":
                    checksum = int(os.getxattr(path, "eos.checksum"), 16)
                else:
                    checksum = None
                # if self._directory.parts[1] == "store":
                #     path = pathlib.Path(
                #         "/store", path.relative_to(config.site.store_path)
                #     )
                self.insert(SampleFile(self, path, size=size, checksum=checksum))


class SampleFromDAS(Sample):
    """Sample given by CMS DAS name."""

    _dasname: str
    _instance: str

    def __init__(
        self,
        name: str,
        type: SampleType,
        parent: "SampleGroup",
        dasname: str,
        instance: str = "",
        tree_name: str = "",
        title: str = "",
        attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialise a sample base class.

        Arguments:
            name: Sample name.
            type: Type of the sample(Data/Simulation/Background/Signal).
            parent: Parent Samplegroup.
            dasname: CMS DAS dataset name.
            instance: DAS instance (default guess from dataset name).
            tree_name: Name of the ROOT TTree.
            title: Name of the sample in ROOT latex format.
            attrs: User defined attributes.
        """
        super().__init__(name, type, parent, tree_name, title, attrs)

        self._dasname = dasname
        if instance:
            self._instance = instance
        elif dasname.endswith("/USER"):
            self._instance = "prod/phys03"
        else:
            self._instance = "prod/phys01"

    def __repr__(self) -> str:

        items = [
            f"path={self.path}",
            f"type={self._type.name}",
            f"dasname={self._dasname}",
            f"instance={self._instance}",
            f"#files={len(self)}",
            f"size={self.size:.2a}",
        ]
        if entries := self.entries:
            items.append(f"entries={entries}")
        for key, value in self.attrs.items():
            items.append(f"{key}={value}")
        return f"SampleFromDAS({', '.join(items)})"

    def get_files(self) -> None:
        """Get the files from DAS and create file structure."""
        cmd = [
            config.bin.dasgoclient,
            "--json",
            f"--query=file dataset={self._dasname} instance={self._instance}",
        ]
        stdout = subprocess.check_output(cmd)  # noqa:S603

        for item in json.loads(stdout):
            for file_item in item["file"]:
                if file_item["name"] not in self:
                    self.insert(
                        SampleFile(
                            self,
                            file_item["name"],
                            size=int(file_item["size"]),
                            entries=int(file_item["nevents"]),
                            checksum=int(file_item["adler32"], 16),
                        )
                    )


class SampleGroup(SampleBase):
    """Sample group."""

    _children: dict[str, SampleBase]

    def __init__(
        self,
        name: str,
        type: SampleType,
        parent: Optional["SampleGroup"] = None,
        tree_name: str = "",
        title: str = "",
        attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialise a sample base class.

        Arguments:
            name: Sample name.
            type: Type of the sample(Data/Simulation/Background/Signal).
            parent: Parent Samplegroup.
            tree_name: Name of the ROOT TTree
            title: Name of the sample in ROOT latex format.
            attrs: User defined attributes.
        """
        super().__init__(name, type, parent, tree_name, title, attrs)
        self._children = {}

    def __repr__(self) -> str:

        items = [
            f"path={self.path}",
            f"type={self._type.name}",
            f"#samples={len(self._children)}",
            f"#files={len(self)}",
            f"size={self.size:.2a}",
        ]
        if entries := self.entries:
            items.append(f"entries={entries}")
        for key, value in self.attrs.items():
            items.append(f"{key}={value}")
        return f"SampleGroup({', '.join(items)})"

    @property
    def children(self) -> Iterator[SampleBase]:
        """Iterator on children samples."""
        return iter(self._children.values())

    def get(self, name: str) -> Optional[SampleBase]:
        """Access a child sample by name."""
        return self._children.get(name)

    def __iter__(self) -> Iterator[SampleFile]:
        """Iterator on contained files."""
        return itertools.chain.from_iterable(iter(s) for s in self._children.values())

    def __len__(self) -> int:
        """Number of contained files."""
        return sum(len(s) for s in self._children.values())

    def __contains__(self, item: Any) -> bool:
        """Path item contained in sample."""
        return any(item in s for s in self._children.values())

    def load(
        self,
        data: Any,
    ) -> None:
        """Load nested samples."""
        if not isinstance(data, list) or any(
            (not isinstance(item, dict) for item in data)
        ):
            log.error("A list of dictionaries is expected.")
            return

        for item in data:
            name = item.get("name")
            if name is None:
                log.error("Item has no name attribute.")
                continue
            if "samples" in item:
                sample_group = SampleGroup(
                    name,
                    SampleGroup.get_type(item, self._type),
                    self,
                    item.get("tree_name", self.tree_name),
                    item.get("title", name),
                    item.get("attributes", {}),
                )
                sample_group.load(item["samples"])
            elif "dasname" in item:
                SampleFromDAS(
                    name,
                    SampleGroup.get_type(item, self._type),
                    self,
                    item["dasname"],
                    item.get("instance"),
                    item.get("tree_name", self.tree_name),
                    item.get("title", name),
                    item.get("attributes", {}),
                )
            elif "directory" in item:
                SampleFromFS(
                    name,
                    SampleGroup.get_type(item, self._type),
                    self,
                    item["directory"],
                    item.get("filter"),
                    item.get("tree_name", self._tree_name),
                    item.get("title", name),
                    item.get("attributes", {}),
                )

    @staticmethod
    def get_type(item: dict[str, Any], default: SampleType) -> SampleType:
        """Get sample type fro dictionary."""
        st = item.get("type", default.name)
        try:
            return SampleType[st.upper()]
        except KeyError:
            log.error("Unexpected SampleType %s", st)
            return SampleType.UNKNOWN


def walk(
    sample: SampleBase, topdown: bool = True
) -> Generator[Tuple[SampleBase, Iterator[Sample], Iterator[SampleGroup]], None, None]:
    """Walk a sample tree."""
    s_iter = (s for s in sample.children if isinstance(s, Sample))
    g_iter = (s for s in sample.children if isinstance(s, SampleGroup))
    if topdown:
        yield sample, s_iter, g_iter
    for s in sample.children:
        yield from walk(s, topdown)
    if not topdown:
        yield sample, s_iter, g_iter


def filter_types(sample: SampleBase, types: SampleTypeSpec = None) -> bool:
    """File samples based on sample type.

    The requested type can be specified by single value or by a container.

    filter_sample(sample, SampleType.DATA)
    filter_sample(sample, [SampleType.Background, SampleType.Signal])
    """
    if types is None:
        return True
    elif isinstance(types, Sequence):
        return sample.type in types
    else:
        return sample.type == types
