import numpy as np
import photonzip
import photonzip.codec.mans as mans
from pathlib import Path
from time import perf_counter
root = Path(__file__).resolve().parents[2]
csv_path = root / "build" / "best_threads.csv"
table = mans.ThreadTable.from_csv(csv_path)
data_path = root / "testdata" / "u2" / "sfc-gi" / "sfc-gi_127x127x127_4096kB.u2"
x = np.fromfile(data_path, dtype=np.uint16).reshape(127, 127, 127)

t0 = perf_counter()
packed = photonzip.compress(
    x,
    codec_options=mans.MansOptions(thread_table=table),
)
t1 = perf_counter()
restored = photonzip.decompress(packed)
y = np.from_dlpack(restored)
t2 = perf_counter()
ratio = x.nbytes / packed.nbytes

print(f"compress: {x.nbytes / (t1 - t0) / 1e6:.2f} MB/s, decompress: {x.nbytes / (t2 - t1) / 1e6:.2f} MB/s")
print(f"csv_path: {csv_path}, thread_rows: {len(table.rows)}")
print(f"ratio: {ratio:.3f}x, restored_type: {type(restored).__name__}")
print(f"is_equal: {np.array_equal(y, x)}")
