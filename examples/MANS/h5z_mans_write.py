from pathlib import Path
import os
from time import perf_counter

root = Path(__file__).resolve().parents[2]
os.environ["HDF5_PLUGIN_PATH"] = str(root / "build" / "bin" / "plugins")
os.environ.setdefault("MANS_THREAD_CSV", str(root / "build" / "best_threads.csv"))

import h5py
import numpy as np
import photonzip.codec.mans as mans

src = root / "testdata" / "u2" / "sfc-gi" / "sfc-gi_127x127x127_4096kB.u2"
dst = root / "build" / "compressed.h5"
x = np.fromfile(src, dtype=np.uint16).reshape(127, 127, 127)
chunks = (16, 16, 16)

t0 = perf_counter()
with h5py.File(dst, "w") as f:
    f.create_dataset(
        "data",
        data=x,
        chunks=chunks,
        compression=mans.H5Z_FILTER_MANS_ID,
        compression_opts=mans.to_hdf5_compression_opts(
            x,
            chunks=chunks,
            options=mans.MansOptions(mode="r"),
        ),
    )
t1 = perf_counter()

print(dst)
print(f"compress: {x.nbytes / (t1 - t0) / 1e6:.2f} MB/s")
