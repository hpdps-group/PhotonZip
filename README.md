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

## 🧾 Citation


- [MANS](https://doi.org/10.1145/3712285.3759825)

## 🕰️ History

- `2026-04-03`: Integrated MANS CPU and GPU paths into the tensor-first Python framework。


## ✅ Tests

Run:

```bash
pytest -q tests/python/test_codecs.py
```
