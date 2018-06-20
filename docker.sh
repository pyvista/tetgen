#!/bin/bash

# builds python wheels on docker container and tests installation

cd /root/source
mkdir wheels  # final directory
for PYBIN in /opt/python/*/bin; do
    # skip Python 3.4 and 3.7
    if [[ $PYBIN =~ .*37.* ]]
    then
    	continue
    elif [[ $PYBIN =~ .*34.* ]]
    then
	continue
    elif [[ $PYBIN =~ .*27.* ]]
    then
	continue
    fi

    pyver="$(cut -d'/' -f4 <<<'/opt/python/cp35-cp35m/bin')"
    echo 'Running for' $pyver
    
    "${PYBIN}/pip" install numpy -q  # required for setup.py
    "${PYBIN}/pip" install cython --upgrade -q

    # build wheel
    "${PYBIN}/python" setup.py -q bdist_wheel

    # test wheel
    wheelfile=$(ls /root/source/dist/*.whl)
    auditwheel repair $wheelfile > /dev/null
    rm dist/*

    wheelfile=$(ls /root/source/wheelhouse/*.whl)
    "${PYBIN}/pip" install $wheelfile -q --no-cache-dir
    mv $wheelfile wheels/

    # pytest doesn't seem to work here
    # "${PYBIN}/pip" install pytest -q
    # "${PYBIN}/python" -m pytest

    # test
    "${PYBIN}/pip" install pyansys -q
    "${PYBIN}/python" tests/test_tetgen.py

done
