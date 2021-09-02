DESCENT
=======
[![Test Status](https://github.com/simonboothroyd/descent/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/descent/actions/workflows/ci.yaml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/SimonBoothroyd/descent.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SimonBoothroyd/descent/context:python)
[![codecov](https://codecov.io/gh/simonboothroyd/descent/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/descent/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Optimize force field parameters against QC data using `pytorch`.

**Warning:** This code is currently experimental and under active development. If you are using this code,
be aware that it is not guaranteed to provide the correct results, and the API can change without notice.

## Installation

The package and its dependencies can be installed into a new conda environment using the `conda` package manager:

```shell
conda env create --name descent --file devtools/conda-envs/meta.yaml
```

## Getting Started

Examples for using this framework can be found in the [`examples`](examples) directory.

### Copyright

Copyright (c) 2021, Simon Boothroyd
