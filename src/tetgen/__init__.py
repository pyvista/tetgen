"""Tetgen module."""

from importlib.metadata import PackageNotFoundError, version

from tetgen.pytetgen import TetGen


try:
    __version__ = version("tetgen")
except PackageNotFoundError:
    __version__ = "unknown"
