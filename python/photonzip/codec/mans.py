from __future__ import annotations

import csv
import os
import signal
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable

from ..codecs import Codec
from ..codec_registry import register_codec
from ..options import CodecOperation, CodecOptions

try:
    _native = import_module("photonzip._native")
except ModuleNotFoundError:
    _native = import_module("_native")

_DEFAULT_THREADS_WARNING_EMITTED = False
H5Z_FILTER_MANS_ID = 32032


def _warn_default_threads_once() -> None:
    global _DEFAULT_THREADS_WARNING_EMITTED
    if _DEFAULT_THREADS_WARNING_EMITTED:
        return
    warnings.warn(
        "No MANS autotune thread table was provided; using the default thread setting (32, 32).",
        RuntimeWarning,
        stacklevel=3,
    )
    _DEFAULT_THREADS_WARNING_EMITTED = True


def _normalize_shape(tensor) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError("Tensor-like input must expose a shape attribute for MANS thread selection.")
    normalized = tuple(int(dim) for dim in shape)
    if not normalized:
        raise ValueError("MANS requires tensor rank to be between 1 and 3.")
    return normalized


def _element_count(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _thread_csv_path_from_env() -> str | Path | None:
    csv_path = os.environ.get("MANS_THREAD_CSV")
    if csv_path:
        return csv_path
    return None


def _normalize_chunk_shape(chunks: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(dim) for dim in chunks)
    if not normalized:
        raise ValueError("MANS HDF5 filter requires a non-empty chunk shape.")
    if any(dim <= 0 for dim in normalized):
        raise ValueError("MANS HDF5 filter chunk dimensions must be positive.")
    return normalized


def _mans_dims_from_shape(shape: Iterable[int]) -> tuple[int, int, int, int]:
    normalized = tuple(int(dim) for dim in shape)
    if len(normalized) == 1:
        return 1, normalized[0], 0, 0
    if len(normalized) == 2:
        return 2, normalized[0], normalized[1], 0

    merged_tail = 1
    for dim in normalized[2:]:
        merged_tail *= dim
    return 3, normalized[0], normalized[1], merged_tail


def _mans_dtype_from_input(value) -> int:
    dtype = getattr(value, "dtype", value)
    itemsize = getattr(dtype, "itemsize", None)
    kind = getattr(dtype, "kind", None)

    if itemsize == 2 and kind == "u":
        return 0
    if itemsize == 4 and kind == "u":
        return 1

    raise TypeError(
        "MANS HDF5 filter only supports unsigned 16-bit or unsigned 32-bit arrays."
    )


@dataclass(frozen=True)
class ThreadConfig:
    chunk_elements: int
    compress_thread: int
    decompress_thread: int
    dims: int


@dataclass(frozen=True)
class SweepRow:
    chunk_elements: int
    dims: int
    mode: str
    threads: int
    throughput_mbps: float


@dataclass(frozen=True)
class ThreadTable:
    rows: tuple[ThreadConfig, ...]

    @classmethod
    def from_response(cls, rows: Iterable[dict]) -> "ThreadTable":
        return cls(
            tuple(
                ThreadConfig(
                    chunk_elements=int(row["chunk_elements"]),
                    compress_thread=int(row["compress_thread"]),
                    decompress_thread=int(row["decompress_thread"]),
                    dims=int(row["dims"]),
                )
                for row in rows
            )
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "ThreadTable":
        csv_path = Path(path)
        rows: list[ThreadConfig] = []
        with csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            first = True
            for cols in reader:
                if not cols:
                    continue
                if first:
                    first = False
                    if "chunk_elements" in cols[0]:
                        continue
                if len(cols) not in (4, 5):
                    continue
                chunk_str = cols[0]
                comp_str = cols[1] if len(cols) == 4 else cols[2]
                decomp_str = cols[2] if len(cols) == 4 else cols[3]
                dims_str = cols[3] if len(cols) == 4 else cols[4]
                try:
                    rows.append(
                        ThreadConfig(
                            chunk_elements=int(chunk_str),
                            compress_thread=int(comp_str),
                            decompress_thread=int(decomp_str),
                            dims=int(dims_str),
                        )
                    )
                except ValueError:
                    continue
        if not rows:
            raise ValueError(f"No valid rows found in MANS thread CSV: {csv_path}")
        return cls(tuple(rows))

    def to_csv(self, path: str | Path) -> Path:
        csv_path = Path(path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["chunk_elements", "compress_thread", "decompress_thread", "dims"])
            for row in self.rows:
                writer.writerow(
                    [row.chunk_elements, row.compress_thread, row.decompress_thread, row.dims]
                )
        return csv_path

    def find_nearest(self, *, chunk_elements: int, dims: int) -> ThreadConfig | None:
        best: ThreadConfig | None = None
        best_diff: int | None = None
        for row in self.rows:
            if row.dims != dims:
                continue
            diff = abs(row.chunk_elements - chunk_elements)
            if best is None or best_diff is None or diff < best_diff or (
                diff == best_diff and row.chunk_elements < best.chunk_elements
            ):
                best = row
                best_diff = diff
        return best


@dataclass(frozen=True)
class AutotuneResult:
    sweep_rows: tuple[SweepRow, ...]
    thread_table: ThreadTable

    @classmethod
    def from_response(cls, result: dict) -> "AutotuneResult":
        sweep_rows = tuple(
            SweepRow(
                chunk_elements=int(row["chunk_elements"]),
                dims=int(row["dims"]),
                mode=str(row["mode"]),
                threads=int(row["threads"]),
                throughput_mbps=float(row["throughput_mbps"]),
            )
            for row in result["sweep_rows"]
        )
        thread_table = ThreadTable.from_response(result["best_configs"])
        return cls(sweep_rows=sweep_rows, thread_table=thread_table)


@dataclass(frozen=True)
class MansOptions(CodecOptions):
    mode: str = "p"
    adm_compress_thread: int | None = None
    adm_decompress_thread: int | None = None
    thread_table: ThreadTable | None = None
    thread_csv_path: str | Path | None = None
    warn_on_default_threads: bool = True

    @property
    def codec(self) -> str:
        return "mans"

    def _resolve_threads(self, *, tensor=None, backend: str = "auto") -> tuple[int, int]:
        if tensor is None:
            raise TypeError("MANS thread resolution requires a tensor-like input shape.")
        return self._resolve_threads_for_shape(_normalize_shape(tensor), backend=backend)

    def _resolve_threads_for_shape(self, shape: Iterable[int], *, backend: str = "auto") -> tuple[int, int]:
        if self.adm_compress_thread is not None and self.adm_decompress_thread is not None:
            return int(self.adm_compress_thread), int(self.adm_decompress_thread)

        table = self.thread_table
        if table is None and self.thread_csv_path is not None:
            table = ThreadTable.from_csv(self.thread_csv_path)

        if table is not None:
            normalized_shape = tuple(int(dim) for dim in shape)
            chosen = table.find_nearest(
                chunk_elements=_element_count(normalized_shape),
                dims=min(len(normalized_shape), 3),
            )
            if chosen is not None:
                return chosen.compress_thread, chosen.decompress_thread

        normalized_backend = str(backend).lower()
        if self.warn_on_default_threads and normalized_backend not in ("cuda", "nv"):
            _warn_default_threads_once()
        compress_thread = 32 if self.adm_compress_thread is None else int(self.adm_compress_thread)
        decompress_thread = 32 if self.adm_decompress_thread is None else int(self.adm_decompress_thread)
        return compress_thread, decompress_thread

    def to_codec_params(self, *, tensor=None, backend: str = "auto") -> list[int]:
        if self.mode == "p":
            mans_mode = 0
        elif self.mode == "r":
            mans_mode = 1
        else:
            raise ValueError(f"Unsupported MANS mode: {self.mode}")

        compress_thread, decompress_thread = self._resolve_threads(tensor=tensor, backend=backend)
        return [mans_mode, compress_thread, decompress_thread]


def to_hdf5_compression_opts(
    tensor,
    *,
    chunks: Iterable[int],
    options: MansOptions | None = None,
) -> tuple[int, ...]:
    chunk_shape = _normalize_chunk_shape(chunks)
    mans_options = MansOptions(thread_csv_path=_thread_csv_path_from_env()) if options is None else options
    if mans_options.thread_table is None and mans_options.thread_csv_path is None:
        mans_options = MansOptions(
            mode=mans_options.mode,
            adm_compress_thread=mans_options.adm_compress_thread,
            adm_decompress_thread=mans_options.adm_decompress_thread,
            thread_table=mans_options.thread_table,
            thread_csv_path=_thread_csv_path_from_env(),
            warn_on_default_threads=mans_options.warn_on_default_threads,
        )

    if len(chunk_shape) > 3:
        effective_shape = chunk_shape[:2] + (_element_count(chunk_shape[2:]),)
    else:
        effective_shape = chunk_shape

    if mans_options.mode == "p":
        mans_mode = 0
    elif mans_options.mode == "r":
        mans_mode = 1
    else:
        raise ValueError(f"Unsupported MANS mode: {mans_options.mode}")

    compress_thread, decompress_thread = mans_options._resolve_threads_for_shape(effective_shape)
    dims, nx, ny, nz = _mans_dims_from_shape(chunk_shape)
    dtype = _mans_dtype_from_input(tensor)
    return (
        0,
        dtype,
        compress_thread,
        decompress_thread,
        mans_mode,
        dims,
        nx,
        ny,
        nz,
    )


@dataclass(frozen=True)
class MansAutotuneOptions(CodecOperation):
    data_size_mb_list: tuple[float, ...] = (
        4.0 / 1024.0,
        8.0 / 1024.0,
        16.0 / 1024.0,
        32.0 / 1024.0,
        64.0 / 1024.0,
        128.0 / 1024.0,
        256.0 / 1024.0,
        512.0 / 1024.0,
        1.0,
        4.0,
        16.0,
        256.0,
    )
    threads_min: int = 1
    threads_max: int = 0
    stride: int = 8
    iter: int = 10
    verbose: bool = False

    @property
    def codec(self) -> str:
        return "mans"

    @property
    def operation(self) -> str:
        return "autotune"

    def to_request(self) -> dict:
        return {
            "data_size_mb_list": [float(value) for value in self.data_size_mb_list],
            "threads_min": int(self.threads_min),
            "threads_max": int(self.threads_max),
            "stride": int(self.stride),
            "iter": int(self.iter),
            "verbose": bool(self.verbose),
        }


def autotune(operation_options: MansAutotuneOptions) -> AutotuneResult:
    if not hasattr(operation_options, "codec"):
        raise TypeError("operation_options must provide codec.")
    if operation_options.codec != "mans":
        raise ValueError(f"Expected MANS operation options, got codec={operation_options.codec!r}.")
    if not hasattr(operation_options, "operation"):
        raise TypeError("operation_options must provide operation.")
    if not hasattr(operation_options, "to_request"):
        raise TypeError("operation_options must provide to_request().")

    repo_root = Path(__file__).resolve().parents[3]
    autotune_bin = repo_root / "build" / "bin" / "cpu" / "cpu_mans_autotune"
    if autotune_bin.exists():
        return _autotune_via_subprocess(autotune_bin, operation_options)

    result = _native.invoke_codec(
        operation_options.codec,
        operation_options.operation,
        request=operation_options.to_request(),
    )
    return AutotuneResult.from_response(result)


class MansCodec(Codec):
    @property
    def name(self) -> str:
        return "mans"

    def compress(
        self,
        tensor,
        *,
        backend="auto",
        codec_params=None,
        codec_options=None,
    ):
        if codec_params is not None and codec_options is not None:
            raise TypeError("Pass either codec_params or codec_options, not both.")
        if codec_options is None:
            codec_options = MansOptions()
        if not hasattr(codec_options, "to_codec_params") or not hasattr(codec_options, "codec"):
            raise TypeError("codec_options must provide codec and to_codec_params().")
        return _native.compress_tensor(
            self.name,
            tensor,
            backend=backend,
            codec_params=codec_options.to_codec_params(tensor=tensor, backend=backend)
            if codec_params is None else codec_params,
        )

    def decompress(self, data, *, backend="auto", **_kwargs):
        return _native.decompress_tensor(data, backend=backend)

    def invoke(self, operation, request=None):
        request = {} if request is None else dict(request)
        if operation == "autotune":
            defaults = MansAutotuneOptions()
            options = MansAutotuneOptions(
                data_size_mb_list=tuple(float(value) for value in request.get("data_size_mb_list", defaults.data_size_mb_list)),
                threads_min=int(request.get("threads_min", defaults.threads_min)),
                threads_max=int(request.get("threads_max", defaults.threads_max)),
                stride=int(request.get("stride", defaults.stride)),
                iter=int(request.get("iter", defaults.iter)),
                verbose=bool(request.get("verbose", defaults.verbose)),
            )
            return _encode_autotune_result(autotune(options))
        return _native.invoke_codec(self.name, operation, request=request)


def _encode_autotune_result(result: AutotuneResult) -> dict:
    return {
        "sweep_rows": [
            {
                "chunk_elements": row.chunk_elements,
                "dims": row.dims,
                "mode": row.mode,
                "threads": row.threads,
                "throughput_mbps": row.throughput_mbps,
            }
            for row in result.sweep_rows
        ],
        "best_configs": [
            {
                "chunk_elements": row.chunk_elements,
                "compress_thread": row.compress_thread,
                "decompress_thread": row.decompress_thread,
                "dims": row.dims,
            }
            for row in result.thread_table.rows
        ],
    }


def _autotune_via_subprocess(
    autotune_bin: Path, operation_options: MansAutotuneOptions
) -> AutotuneResult:
    with tempfile.TemporaryDirectory(prefix="photonzip-mans-autotune-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        sweep_csv = tmpdir_path / "thread_sweep.csv"
        best_csv = tmpdir_path / "best_threads.csv"
        cmd = [
            str(autotune_bin),
            "--data-size-mb-list",
            ",".join(str(float(value)) for value in operation_options.data_size_mb_list),
            "--threads-min",
            str(int(operation_options.threads_min)),
            "--threads-max",
            str(int(operation_options.threads_max)),
            "--stride",
            str(int(operation_options.stride)),
            "--iter",
            str(int(operation_options.iter)),
            "--csv",
            str(sweep_csv),
            "--out",
            str(best_csv),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=None if operation_options.verbose else subprocess.DEVNULL,
            stderr=None if operation_options.verbose else subprocess.DEVNULL,
            cwd=autotune_bin.parent,
            env=os.environ.copy(),
        )
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
            raise
        if rc != 0:
            raise RuntimeError(f"MANS autotune failed with exit code {rc}.")
        return _load_autotune_result_from_csvs(sweep_csv, best_csv)


def _load_autotune_result_from_csvs(sweep_csv: Path, best_csv: Path) -> AutotuneResult:
    sweep_rows: list[SweepRow] = []
    with sweep_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sweep_rows.append(
                SweepRow(
                    chunk_elements=int(row["chunk_elements"]),
                    dims=int(row["dims"]),
                    mode=str(row["mode"]),
                    threads=int(row["threads"]),
                    throughput_mbps=float(row["throughput_mbps"]),
                )
            )

    return AutotuneResult(
        sweep_rows=tuple(sweep_rows),
        thread_table=ThreadTable.from_csv(best_csv),
    )


register_codec(MansCodec())
