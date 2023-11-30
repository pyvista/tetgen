#!/bin/bash
set -e

tmpdir=$(mktemp -d)
mkdir -p out
cp bar3.poly $tmpdir
python generate_background_mesh.py $tmpdir/bar3.poly
./tetgen -pqm $tmpdir/bar3.poly
cp $tmpdir/bar3.1.* out
