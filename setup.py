import os
import numpy
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from io import open as io_open


# Version from file
__version__ = None
version_file = os.path.join(os.path.dirname(__file__), 'tetgen', '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())


setup(
    name='tetgen',
    packages = ['tetgen'],
    version=__version__,
   description='Python interface to pytetgen',
   long_description=open('README.rst').read(),

    author='Alex Kaszynski',
    author_email='akascap@gmail.com',
   url = 'https://github.com/akaszynski/tetgen',

   license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],


    # Build cython modules
    cmdclass={'build_ext': build_ext},
    ext_modules = [Extension("tetgen._tetgen", 
                             ['tetgen/cython/tetgen/_tetgen.pyx', 
                              'tetgen/cython/tetgen/tetgen.cxx', 
                              'tetgen/cython/tetgen/predicates.cxx',
                              'tetgen/cython/tetgen/tetgen_wrap.cxx'],
                             language='c++',
                             extra_compile_args=["-O3"],
                             define_macros=[('TETLIBRARY', None)]),
                   ],

    keywords='TetGen',
                           
    include_dirs=[numpy.get_include()],
                  
    # Might work with earlier versions
    install_requires=['numpy>1.9.3', 'pymeshfix']

)
