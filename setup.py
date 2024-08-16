"""Setup for tetgen."""

from io import open as io_open
import os

from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup

# Version from file
__version__ = None
version_file = os.path.join(os.path.dirname(__file__), "tetgen", "_version.py")
with io_open(version_file, mode="r") as fd:
    exec(fd.read())


# compiler args
if os.name == "nt":  # windows
    extra_compile_args = ["/openmp", "/O2", "/w", "/GS"]
elif os.name == "posix":  # linux org mac os
    extra_compile_args = ["-std=gnu++11", "-O3", "-w"]
else:
    raise Exception(f"Unsupported OS {os.name}")


setup(
    name="tetgen",
    packages=["tetgen"],
    version=__version__,
    description="Python interface to tetgen",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    author="PyVista Developers",
    author_email="info@pyvista.org",
    url="https://github.com/pyvista/tetgen",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    # Build cython modules
    ext_modules=cythonize(
        [
            Extension(
                "tetgen._tetgen",
                [
                    "tetgen/cython/tetgen/_tetgen.pyx",
                    "tetgen/cython/tetgen/tetgen.cxx",
                    "tetgen/cython/tetgen/predicates.cxx",
                    "tetgen/cython/tetgen/tetgen_wrap.cxx",
                ],
                language="c++",
                extra_compile_args=extra_compile_args,
                include_dirs=[np.get_include()],
                define_macros=[("TETLIBRARY", None)],
            ),
        ],
    ),
    python_requires=">=3.9",
    keywords="TetGen",
    install_requires=["numpy>=2,<3", "pyvista>=0.31.0"],
)
