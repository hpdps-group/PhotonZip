import numpy as np
import photonzip
import torch
from pathlib import Path

data_path = Path(__file__).resolve().parents[2] / "testdata/u2/exafel/exafel_59200x388_16384kB.u2"
x = np.fromfile(data_path, dtype=np.uint16).reshape(128, 256, 256)

packed = photonzip.compress(
    x,
    codec_options=photonzip.mans.MansOptions(),
)
y = photonzip.decompress(packed)
ratio = x.nbytes / packed.nbytes

print(f"ratio: {ratio:.3f}x, device: {y.device}, shape: {tuple(y.shape)}")
print(f"is_equal: {np.array_equal(y.cpu().numpy(), x)}")
