"""Setup for tetgen"""
import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from io import open as io_open

try:
    import numpy
except:
    raise ImportError('Please install numpy first')

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# Version from file
__version__ = None
version_file = os.path.join(os.path.dirname(__file__), 'tetgen', '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())

# for: the cc1plus: warning: command line option '-Wstrict-prototypes'
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process:
        try:
            del builtins.__NUMPY_SETUP__
        except AttributeError:
            pass
        import numpy
        self.include_dirs.append(numpy.get_include())

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        _build_ext.build_extensions(self)

# compiler args
macros = []
if os.name == 'nt':  # windows
    extra_compile_args = ['/openmp', '/O2', '/w', '/GS']
elif os.name == 'posix':  # linux org mac os
    extra_compile_args = ['-std=gnu++11', '-O3', '-w']
else:
    raise Exception('Unsupported OS %s' % os.name)


setup(
    name='tetgen',
    packages = ['tetgen'],
    version=__version__,
    description='Python interface to tetgen',
    long_description=open('README.rst').read(),

    author='Alex Kaszynski',
    author_email='akascap@gmail.com',
    url = 'https://github.com/pyvista/tetgen',

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # Build cython modules
    cmdclass={'build_ext': build_ext},
    ext_modules = [Extension("tetgen._tetgen", 
                             ['tetgen/cython/tetgen/_tetgen.pyx', 
                              'tetgen/cython/tetgen/tetgen.cxx', 
                              'tetgen/cython/tetgen/predicates.cxx',
                              'tetgen/cython/tetgen/tetgen_wrap.cxx'],
                             language='c++',
                             extra_compile_args=extra_compile_args,
                             define_macros=[('TETLIBRARY', None)]),
                   ],

    keywords='TetGen',

    install_requires=['numpy>1.9.3',
                      'pyvista>=0.19.0']
)
