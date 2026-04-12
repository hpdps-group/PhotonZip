from pathlib import Path
import os
from time import perf_counter

root = Path(__file__).resolve().parents[2]
os.environ["HDF5_PLUGIN_PATH"] = str(root / "build" / "bin" / "plugins")
os.environ.setdefault("MANS_THREAD_CSV", str(root / "build" / "best_threads.csv"))

import h5py
import numpy as np

src_h5 = root / "build" / "compressed.h5"
src_raw = root / "testdata" / "u2" / "sfc-gi" / "sfc-gi_127x127x127_4096kB.u2"
dst_raw = root / "build" / "restored.u2"

t0 = perf_counter()
with h5py.File(src_h5, "r") as f:
    x = f["data"][:]
t1 = perf_counter()

x.astype(np.uint16, copy=False).tofile(dst_raw)
ok = np.array_equal(x.reshape(-1), np.fromfile(src_raw, dtype=np.uint16))

print(dst_raw)
print(f"decompress: {x.nbytes / (t1 - t0) / 1e6:.2f} MB/s")
print(f"is_equal: {ok}")
