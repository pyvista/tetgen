#!/bin/bash
set -e
tetgen_dir=../tetgen/cython/tetgen

c++ -O0 -o $tetgen_dir/predicates.o -c $tetgen_dir/predicates.cxx
c++ -O3 -o tetgen $tetgen_dir/tetgen.cxx $tetgen_dir/predicates.o -lm