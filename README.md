<h1 align="center">DESCENT</h1>

<p align="center">Optimize force field parameters against reference data</p>

<p align="center">
  <a href="https://github.com//actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/SimonBoothroyd/descent/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/SimonBoothroyd/descent/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/SimonBoothroyd/descent/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

---

The `descent` framework aims to offer a modern API for training classical force field parameters (either from a 
traditional format such as SMIRNOFF or from some ML model) against reference data using `pytorch`.

This framework benefited hugely from [ForceBalance](https://github.com/leeping/forcebalance), and a significant
number of learning from that project, and from Lee-Ping, have influenced the design of this one.

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

## Copyright

Copyright (c) 2023, Simon Boothroyd
