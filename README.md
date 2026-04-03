# 📦 PhotonZip

PhotonZip is a tensor-first Python wrapper around vendored lossless codecs.

The current backend is [MANS](https://github.com/hpdps-group/MANS), exposed through a small high-level API:

- `photonzip.compress(...)`
- `photonzip.decompress(...)`
- `photonzip.mans.autotune(...)`

Internally, the Python layer is DLPack-based, and `decompress(...)` returns a `torch.Tensor`.

## 🚀 Clone

```bash
git clone --recurse-submodules https://github.com/hpdps-group/PhotonZip.git
cd PhotonZip
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## 🔧 Install

```bash
python3 -m pip install .
```

For local development:

```bash
python3 -m pip install -e . --no-deps
```

## 📋 Requirements

- Python 3.9+
- A working C++ toolchain
- CUDA toolkit if you want `backend="cuda"`
- `torch` installed in your Python environment

## 🧪 Examples

See [`examples/python`](./examples/python/):

- `cpu_roundtrip.py`
- `cpu_multidim.py`
- `cuda_host_input.py`
- `cuda_device_input.py`
- `autotune_to_csv.py`
- `compress_with_csv.py`

These examples cover:

- CPU round-trip
- multi-dimensional tensors
- CUDA host input
- CUDA device input
- MANS autotune to CSV
- compression with a saved thread CSV

Each example uses the repository dataset:

- `testdata/u2/exafel/exafel_59200x388_16384kB.u2`

By default, `MansOptions()` uses `mode="p"`.
Use `mode="r"` only when you explicitly want MANS R-mode.

## 🧾 Citation


- [MANS](https://doi.org/10.1145/3712285.3759825)

## 🕰️ History

- `2026-04-03`: Integrated MANS CPU and GPU paths into the tensor-first Python framework。


## ✅ Tests

Run:

```bash
pytest -q tests/python/test_codecs.py
```
