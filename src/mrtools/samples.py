    """Samples for ModernROOTTools.
    """
    import abc
    import collections
    import pathlib
    from  typing import Iterator

    from datasize import DataSize

    class SampleABC(collections.abc.Sized, collections.abc.Iterable):
        """Base class for all types of sample."""
        
        @abc.abstractmethod
        def __str__(self) -> str:
            """Pathname of the sample."""
            pass

        @property
        def path(self) -> pathlib.PurePath:
            """Pathname a pure path object."""
            return pathlib.PurePath(str(self))

        @property
        def stub(self) -> str:
            """Pathname translated to stub without /."""
            return str(self)[1:].replace("/", "_")

        @property
        @abc.abstractmethod
        def tree_name(self) -> str:
            """Name of ROOT TTree."""
            pass

        @abc.abstractmethod
        def __iter__(self) -> Iterator["File"]:
            """Iterator on contained files."""
            pass

        @abc.abstractmethod
        def __len__(self) -> int:
            """Number on contained files."""
            pass

        @property
        @abc.abstractmethod
        def entries(self) -> int:
            pass

        @property
        @abc.abstractmethod
        def size(self) -> DataSize:
            pass

        @abc.abstractmethod
        def samples(self) -> Iterator["SampleABC"]:
            pass





