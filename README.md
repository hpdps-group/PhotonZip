# PhotonZip

PhotonZip is a tensor-first Python wrapper around vendored lossless codecs.  
The current backend is [MANS](https://github.com/hpdps-group/MANS), exposed through a simple API:

- `photonzip.compress(...)`
- `photonzip.decompress(...)`
- `photonzip.mans.autotune(...)`

The Python layer is DLPack-based internally, and `decompress(...)` returns a `torch.Tensor`.

## Clone

```bash
git clone --recurse-submodules https://github.com/hpdps-group/PhotonZip.git
cd PhotonZip
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Install

```bash
python3 -m pip install .
```

For local development:

```bash
python3 -m pip install -e . --no-deps
```

## Requirements

- Python 3.9+
- A working C++ toolchain
- CUDA toolkit if you want `backend="cuda"`
- `torch` installed in your Python environment

## Examples

See [`examples`](./examples/):

Each example uses the repository dataset:

- `testdata/u2/exafel/exafel_59200x388_16384kB.u2`


## Notes

- PhotonZip vendors MANS under `3rdparty/lossless/MANS`.
- The Python package no longer relies on installing a separate `libgpu_ans.so` beside the wheel; the CUDA path is linked in at the lower level.
