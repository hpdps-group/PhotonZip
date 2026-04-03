import numpy as np
import photonzip
import torch
from pathlib import Path

csv_path = Path(__file__).with_name("best_threads.csv")
table = photonzip.mans.ThreadTable.from_csv(csv_path)
data_path = Path(__file__).resolve().parents[2] / "testdata/u2/exafel/exafel_59200x388_16384kB.u2"
x = np.fromfile(data_path, dtype=np.uint16)

packed = photonzip.compress(
    x,
    codec_options=photonzip.mans.MansOptions(thread_table=table),
)
y = photonzip.decompress(packed)
ratio = x.nbytes / packed.nbytes

print(f"csv_path: {csv_path}, thread_rows: {len(table.rows)}")
print(f"ratio: {ratio:.3f}x, device: {y.device}")
print(f"is_equal: {np.array_equal(y.cpu().numpy(), x)}")
