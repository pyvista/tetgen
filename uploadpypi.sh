#!/bin/bash

# don't upload eggs
rm dist/*egg

# upload to PyPI
twine upload dist/*
