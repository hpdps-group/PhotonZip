from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from time import perf_counter

import numpy as np
import photonzip
import photonzip.codec.mans as mans
import torch

try:
    from photonzip import _native
except ImportError:
    import _native


MAGIC = b"PZC1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREAD_CSV = REPO_ROOT / "build" / "best_threads.csv"
LOSSLESS_CODEC = "mans"
QUALITY_LEVEL_ERROR_BOUNDS = {
    "lossless": 0.0,
    "high": 1.0,
    "low": 8.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple PhotonZip CLI")
    parser.add_argument("--mode", required=True, choices=("compress", "decompress", "roundtrip"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dims", nargs="+", type=int, help="Input tensor shape, e.g. --dims 127 127 127")
    parser.add_argument("--dtype", choices=("uint16", "uint32"))
    level_group = parser.add_argument_group("compression levels")
    level_group.add_argument(
        "--quality-level",
        dest="quality_level",
        default="lossless",
        choices=tuple(QUALITY_LEVEL_ERROR_BOUNDS),
        help="Quality level: lossless, high, or low.",
    )
    level_group.add_argument("--throughput-level", choices=("high", "low"), help="Throughput level.")
    level_group.add_argument("--ratio-level", choices=("high", "low"), help="Compression-ratio level.")
    parser.add_argument("--level", dest="quality_level", choices=tuple(QUALITY_LEVEL_ERROR_BOUNDS), help=argparse.SUPPRESS)
    parser.add_argument("--fidelity", dest="quality_level", choices=tuple(QUALITY_LEVEL_ERROR_BOUNDS), help=argparse.SUPPRESS)
    parser.add_argument("--speed", dest="throughput_level", choices=("high", "low"), help=argparse.SUPPRESS)
    parser.add_argument("--ratio", dest="ratio_level", choices=("high", "low"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--codec",
        choices=("mans",),
        help="Force a codec instead of selecting one from the quality level.",
    )
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode in ("compress", "roundtrip"):
        if not args.dims:
            raise ValueError("--dims is required for compress and roundtrip modes.")
        if args.dtype is None:
            raise ValueError("--dtype is required for compress and roundtrip modes.")
        if args.quality_level != "lossless":
            error_bound = QUALITY_LEVEL_ERROR_BOUNDS[args.quality_level]
            raise NotImplementedError(
                f"Lossy compression is not wired yet for quality_level={args.quality_level!r} "
                f"(preset error_bound={error_bound})."
            )
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--backend cuda was requested, but torch.cuda.is_available() is False.")


def read_raw_array(path: Path, dtype_name: str, dims: list[int]) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    shape = tuple(int(dim) for dim in dims)
    array = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(
            f"Input element count mismatch: file contains {array.size} elements, but dims imply {expected}."
        )
    return array.reshape(shape)


def to_backend_tensor(array: np.ndarray, backend: str):
    if backend == "cpu":
        return array
    return torch.from_numpy(array).cuda()


def to_host_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.from_dlpack(value)


def resolve_thread_csv() -> Path | None:
    if DEFAULT_THREAD_CSV.exists():
        return DEFAULT_THREAD_CSV
    return None


def ensure_autotuned_csv() -> Path:
    csv_path = resolve_thread_csv()
    if csv_path is not None:
        print(f"thread_csv: {csv_path}")
        return csv_path

    print(f"thread_csv_missing: {DEFAULT_THREAD_CSV}")
    print("autotune: start")
    result = mans.autotune(
        mans.MansAutotuneOptions(
            data_size_mb_list=(4.0 / 1024.0, 8.0 / 1024.0, 16.0 / 1024.0, 32.0 / 1024.0, 1.0, 4.0),
            threads_min=1,
            threads_max=64,
            iter=3,
            verbose=False,
        )
    )
    DEFAULT_THREAD_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.thread_table.to_csv(DEFAULT_THREAD_CSV)
    print(f"autotune_csv: {DEFAULT_THREAD_CSV}")
    print(f"thread_rows: {len(result.thread_table.rows)}")
    return DEFAULT_THREAD_CSV


def resolve_lossless_mans_mode(args: argparse.Namespace) -> str:
    if args.throughput_level == "high":
        return "p"
    if args.ratio_level == "high":
        return "r"
    return "p"


def make_codec_options(args: argparse.Namespace, tensor) -> tuple[str, object, list[int]]:
    if args.quality_level != "lossless":
        error_bound = QUALITY_LEVEL_ERROR_BOUNDS[args.quality_level]
        raise NotImplementedError(
            f"Lossy compression is not wired yet for quality_level={args.quality_level!r} "
            f"(preset error_bound={error_bound})."
        )

    codec = args.codec or LOSSLESS_CODEC
    if codec != "mans":
        raise ValueError(f"Unsupported lossless codec: {codec!r}.")

    csv_path = ensure_autotuned_csv()
    options = mans.MansOptions(mode=resolve_lossless_mans_mode(args), thread_csv_path=csv_path)
    codec_params = options.to_codec_params(tensor=tensor, backend=args.backend)
    return codec, options, codec_params


def write_container(path: Path, *, codec: str, backend: str, dtype: str, shape: tuple[int, ...], codec_params: list[int], payload: bytes) -> None:
    metadata = {
        "codec": codec,
        "backend": backend,
        "dtype": dtype,
        "shape": [int(dim) for dim in shape],
        "codec_params": [int(value) for value in codec_params],
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Q", len(metadata_bytes)))
        handle.write(metadata_bytes)
        handle.write(payload)


def read_container(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Unsupported file format in {path}.")
        metadata_size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(metadata_size).decode("utf-8"))
        payload = handle.read()
    return metadata, payload


def print_compress_stats(nbytes: int, elapsed: float, packed_nbytes: int) -> None:
    print(f"compress: {nbytes / elapsed / 1e6:.2f} MB/s")
    print(f"ratio: {nbytes / packed_nbytes:.3f}x")


def print_decompress_stats(nbytes: int, elapsed: float) -> None:
    print(f"decompress: {nbytes / elapsed / 1e6:.2f} MB/s")


def print_roundtrip_stats(nbytes: int, compress_elapsed: float, decompress_elapsed: float, packed_nbytes: int, is_equal: bool) -> None:
    print(f"compress: {nbytes / compress_elapsed / 1e6:.2f} MB/s")
    print(f"decompress: {nbytes / decompress_elapsed / 1e6:.2f} MB/s")
    print(f"ratio: {nbytes / packed_nbytes:.3f}x")
    print(f"is_equal: {is_equal}")


def run_compress(args: argparse.Namespace) -> None:
    array = read_raw_array(args.input, args.dtype, args.dims)
    tensor = to_backend_tensor(array, args.backend)
    codec, codec_options, codec_params = make_codec_options(args, tensor)
    t0 = perf_counter()
    packed = photonzip.compress(tensor, codec=codec, backend=args.backend, codec_options=codec_options)
    t1 = perf_counter()
    write_container(
        args.output,
        codec=codec,
        backend=args.backend,
        dtype=args.dtype,
        shape=array.shape,
        codec_params=codec_params,
        payload=packed.to_bytes(),
    )
    print_compress_stats(array.nbytes, t1 - t0, packed.nbytes)


def run_decompress(args: argparse.Namespace) -> None:
    metadata, payload = read_container(args.input)
    packed = _native.compressed_from_bytes(
        metadata["codec"],
        payload,
        metadata["dtype"],
        metadata["shape"],
        metadata["backend"],
        metadata["codec_params"],
    )
    t0 = perf_counter()
    restored = photonzip.decompress(packed, backend=args.backend)
    t1 = perf_counter()
    array = to_host_numpy(torch.from_dlpack(restored) if args.backend == "cuda" else restored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(args.output)
    print_decompress_stats(array.nbytes, t1 - t0)


def run_roundtrip(args: argparse.Namespace) -> None:
    array = read_raw_array(args.input, args.dtype, args.dims)
    tensor = to_backend_tensor(array, args.backend)
    codec, codec_options, codec_params = make_codec_options(args, tensor)
    t0 = perf_counter()
    packed = photonzip.compress(tensor, codec=codec, backend=args.backend, codec_options=codec_options)
    t1 = perf_counter()
    restored = photonzip.decompress(packed, backend=args.backend)
    t2 = perf_counter()
    restored_array = to_host_numpy(torch.from_dlpack(restored) if args.backend == "cuda" else restored)
    write_container(
        args.output,
        codec=codec,
        backend=args.backend,
        dtype=args.dtype,
        shape=array.shape,
        codec_params=codec_params,
        payload=packed.to_bytes(),
    )
    print_roundtrip_stats(array.nbytes, t1 - t0, t2 - t1, packed.nbytes, np.array_equal(restored_array, array))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.mode == "compress":
        run_compress(args)
        return
    if args.mode == "decompress":
        run_decompress(args)
        return
    run_roundtrip(args)


if __name__ == "__main__":
    main()
