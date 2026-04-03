import numpy as np
import torch
import photonzip
from pathlib import Path

data_path = Path(__file__).resolve().parents[2] / "testdata/u2/exafel/exafel_59200x388_16384kB.u2"
x = torch.from_numpy(np.fromfile(data_path, dtype=np.uint16)).cuda()

packed = photonzip.compress(
    x,
    backend="cuda",
    codec_options=photonzip.mans.MansOptions(),
)
y = photonzip.decompress(packed)
ratio = (x.element_size() * x.numel()) / packed.nbytes

print(f"ratio: {ratio:.3f}x, input_device: {x.device}, output_device: {y.device}")
print(f"is_equal: {torch.equal(y, x)}")
