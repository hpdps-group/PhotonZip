import numpy as np
import torch
import photonzip
import photonzip.codec.mans as mans
from pathlib import Path
from time import perf_counter

root = Path(__file__).resolve().parents[2]
data_path = root / "testdata" / "u2" / "sfc-gi" / "sfc-gi_127x127x127_4096kB.u2"
x = torch.from_numpy(np.fromfile(data_path, dtype=np.uint16).reshape(127, 127, 127)).cuda()

t0 = perf_counter()
packed = photonzip.compress(
    x,
    backend="cuda",
    codec_options=mans.MansOptions(),
)
t1 = perf_counter()
restored = photonzip.decompress(packed)
y = torch.from_dlpack(restored)
t2 = perf_counter()
ratio = (x.element_size() * x.numel()) / packed.nbytes

print(f"compress: {(x.element_size() * x.numel()) / (t1 - t0) / 1e6:.2f} MB/s, decompress: {(x.element_size() * x.numel()) / (t2 - t1) / 1e6:.2f} MB/s")
print(f"ratio: {ratio:.3f}x, input_device: {x.device}, output_device: {y.device}")
print(f"is_equal: {torch.equal(y, x)}")
