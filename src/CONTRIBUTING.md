### Contributing to the C Extension

This repository interfaces with `tetgen.cxx` via
[nanobind](https://github.com/wjakob/nanobind) to efficiently generate C
extensions.


#### Emacs configuration

If using emacs and helm, generate the project configuration files using
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. Here's a sample configuration for
``nixos``:


```
rm -rf CMakeCache.txt CMakeFiles compile_commands.json build  # optional
pip install nanobind
export NANOBIND_INCLUDE=$(python -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))")
cmake -Dnanobind_DIR=$NANOBIND_INCLUDE \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
```
