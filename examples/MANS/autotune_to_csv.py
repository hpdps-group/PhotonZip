import numpy as np
import photonzip
import torch
from pathlib import Path

result = photonzip.mans.autotune(
    photonzip.mans.MansAutotuneOptions(
        data_size_mb_list=(4.0,), dims_list=(1,), threads_min=1, threads_max=64, verbose=False
    )
)
csv_path = Path(__file__).with_name("best_threads.csv")
result.thread_table.to_csv(csv_path)
data_path = Path(__file__).resolve().parents[2] / "testdata/u2/exafel/exafel_59200x388_16384kB.u2"
x = np.fromfile(data_path, dtype=np.uint16)
opts = photonzip.mans.MansOptions(thread_table=result.thread_table)

packed = photonzip.compress(x, codec_options=opts)
y = photonzip.decompress(packed)
ratio = x.nbytes / packed.nbytes

print(f"thread_rows: {len(result.thread_table.rows)}, ratio: {ratio:.3f}x, device: {y.device}")
print(f"csv_path: {csv_path}")
print(f"is_equal: {np.array_equal(y.cpu().numpy(), x)}")
