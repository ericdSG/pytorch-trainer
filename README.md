# :fire: PyTorch Training Template

![python](https://img.shields.io/badge/python-3.9-blue.svg)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Spec.](https://docs.google.com/document/d/1O5z5TiHfbNJMPLjjezeKoCgch9f_ChQlss5xfnQr3Ao/edit)

# :gear: Installation

Clone the repo, navigate to the top-level directory, and:

1. Initialize submodules
```bash
git submodule update --init
git lfs install && git lfs pull
```

1. Create a new conda environment `trainer` with dependencies from `environment.yml`
```bash
mamba env create && mamba activate trainer
```

1. Enable pre-commit hooks
```bash
pre-commit install --install-hooks
```

# TODO

- [] Add TensorBoard support
- [] Add `DistributedDataParallel` support
- [] Containerize with Docker
