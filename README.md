# The EigenIO library: binary IO for Eigen3 matrices

A full template library for binary IO of Eigen3 matrices and vectors. It was
designed to be as simple as possible and not rely on any outside dependencies
outside Eigen. It is not meant to provide a reliable long-term
cross-platform file format.

Supports both dense and sparse matrices and any size of float and integer types.

## File Format specification
Described in the header of EigenIO.h

## Current Limitations
* Only column-major and sparse CSC matrices are supported (no row-major support)
* Endianness is not accounted for

## Requirements
### C++
* Eigen3 (see http://eigen.tuxfamily.org)
* A C++14 compliant compiler

### Python
* Python3.5
* Numpy
* Scipy (for sparse matrix support)


## Installing
### C++ header:
` mkdir build && cd build && cmake .. && make install`

### Python module
Use the setup.py script, for help type `python3 setup.py install --help`

## Building and running tests
Currently only the C++ version has tests, which can be run using the following command:
` mkdir build && cd build && cmake .. -DBUILD_TEST=ON && make && make test`

These are built using the Catch2 library, see https://github.com/catchorg/Catch2
