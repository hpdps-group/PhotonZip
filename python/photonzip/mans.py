from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .codecs import invoke_codec
from .options import CodecOperation, CodecOptions

_DEFAULT_THREADS_WARNING_EMITTED = False


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

    def _resolve_threads(self, *, tensor=None) -> tuple[int, int]:
        if self.adm_compress_thread is not None and self.adm_decompress_thread is not None:
            return int(self.adm_compress_thread), int(self.adm_decompress_thread)

        table = self.thread_table
        if table is None and self.thread_csv_path is not None:
            table = ThreadTable.from_csv(self.thread_csv_path)

        if table is not None:
            if tensor is None:
                raise TypeError("MANS thread-table selection requires the tensor input shape.")
            shape = _normalize_shape(tensor)
            chosen = table.find_nearest(chunk_elements=_element_count(shape), dims=len(shape))
            if chosen is not None:
                return chosen.compress_thread, chosen.decompress_thread

        if self.warn_on_default_threads:
            _warn_default_threads_once()
        compress_thread = 32 if self.adm_compress_thread is None else int(self.adm_compress_thread)
        decompress_thread = 32 if self.adm_decompress_thread is None else int(self.adm_decompress_thread)
        return compress_thread, decompress_thread

    def to_codec_params(self, *, tensor=None) -> list[int]:
        if self.mode == "p":
            mans_mode = 0
        elif self.mode == "r":
            mans_mode = 1
        else:
            raise ValueError(f"Unsupported MANS mode: {self.mode}")

        compress_thread, decompress_thread = self._resolve_threads(tensor=tensor)
        return [
            mans_mode,
            compress_thread,
            decompress_thread,
        ]


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
    dims_list: tuple[int, ...] = (1, 2, 3)
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
            "dims_list": [int(value) for value in self.dims_list],
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
    result = invoke_codec(
        operation_options.codec,
        operation_options.operation,
        request=operation_options.to_request(),
    )
    return AutotuneResult.from_response(result)
