DESCENT
=======
[![Test Status](https://github.com/simonboothroyd/descent/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/descent/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/simonboothroyd/descent/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/descent/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `descent` framework aims to offer a modern API for training classical force field parameters (either from a 
traditional format such as SMIRNOFF or from some ML model) against reference data using `pytorch`.

***Warning**: This code is currently experimental and under active development. If you are using this it, please be 
aware that it is not guaranteed to provide correct results, the documentation and testing maybe be incomplete, and the
API can change without notice.*

## Installation

This package can be installed using `conda` (or `mamba`, a faster version of `conda`):

```shell
mamba install -c conda-forge descent
```

The example notebooks further require you install `jupyter`:

```shell
mamba install -c conda-forge jupyter
```

## Getting Started

To get started, see the [examples](examples).

## Development

A development conda environment can be created and activated by running:

```shell
make env
conda activate descent
```

The environment will include all example and development dependencies, including linters and testing apparatus.

Unit / example tests can be run using:

```shell
make test
```

The codebase can be formatted by running:

```shell
make format
```

or checked for lint with:

```shell
make lint
```

## License

The main package is release under the [MIT license](LICENSE). 

## Copyright

Copyright (c) 2023, Simon Boothroyd
