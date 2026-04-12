import pathlib
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest


sys.path.insert(0, "/root/workspace/PhotonZip/build/bindings/python")
sys.path.insert(0, "/root/workspace/PhotonZip/python")

import photonzip
import photonzip.codec.mans as mans


DATASET = pathlib.Path(
    "/root/workspace/PhotonZip/3rdparty/lossless/MANS/testdata/u2/exafel/exafel_59200x388_64kB.u2"
)

CPU_DEVICE = int(np.arange(1, dtype=np.uint8).__dlpack_device__()[0])


def _u16():
    return np.fromfile(DATASET, dtype=np.uint16)


def _roundtrip(x, **kwargs):
    packed = photonzip.compress(x, **kwargs)
    return np.from_dlpack(photonzip.decompress(packed))


def _mans(**kwargs):
    return mans.MansOptions(warn_on_default_threads=False, **kwargs)


def _torch():
    return pytest.importorskip("torch")


def _device_code(obj):
    return int(obj.__dlpack_device__()[0])


def _skip_if_cuda_backend_missing(exc: RuntimeError):
    message = str(exc)
    if "NVIDIA backend was NOT compiled" in message:
        pytest.skip("MANS CUDA backend was not compiled")
    if "Unsupported input memory kind." in message:
        pytest.skip("Direct CUDA tensor input is not available in this build")
    raise exc


def _run_python(code: str) -> str:
    wrapped = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, "/root/workspace/PhotonZip/build/bindings/python")
        sys.path.insert(0, "/root/workspace/PhotonZip/python")
        import photonzip.codec.mans as mans
        {code}
    """)
    result = subprocess.run([sys.executable, "-c", wrapped], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    if result.stdout.startswith("SKIP:"):
        pytest.skip(result.stdout.strip()[5:])
    raise AssertionError(result.stderr or result.stdout or "python snippet failed")


def test_numpy_cpu_1d():
    x = _u16()
    y = _roundtrip(x, codec_options=_mans())
    assert y.tobytes() == x.tobytes()


def test_numpy_cpu_2d():
    x = _u16().reshape(128, 256)
    y = _roundtrip(x, codec_options=_mans())
    assert y.shape == x.shape and y.tobytes() == x.tobytes()


def test_numpy_cpu_3d():
    x = _u16().reshape(16, 16, 128)
    y = _roundtrip(x, codec_options=_mans())
    assert y.shape == x.shape and y.tobytes() == x.tobytes()


def test_mans_autotune_smoke():
    result = mans.autotune(mans.MansAutotuneOptions(
        data_size_mb_list=(4.0 / 1024.0,), threads_min=1, threads_max=1, stride=1, iter=1
    ))
    assert len(result.thread_table.rows) >= 1 and len(result.sweep_rows) >= 1


def test_mans_thread_table_from_csv(tmp_path):
    p = tmp_path / "best_threads.csv"
    p.write_text("chunk_elements,compress_thread,decompress_thread,dims\n1024,4,5,1\n4096,8,9,1\n")
    cfg = mans.ThreadTable.from_csv(p).find_nearest(chunk_elements=2048, dims=1)
    assert (cfg.compress_thread, cfg.decompress_thread) == (4, 5)


def test_mans_default_thread_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = mans.MansOptions().to_codec_params(tensor=np.arange(16, dtype=np.uint16))
    assert params == [0, 32, 32] and caught


def test_torch_cpu_1d():
    torch = _torch()
    x = torch.from_numpy(_u16())
    y = torch.from_dlpack(photonzip.decompress(photonzip.compress(x, codec_options=_mans())))
    assert torch.equal(y, x)


def test_compress_cpu_backend_keeps_payload_on_host():
    packed = photonzip.compress(_u16(), codec_options=_mans())
    assert packed.compressed and _device_code(packed) == CPU_DEVICE


def test_compress_cuda_backend_moves_host_payload_to_device():
    out = _run_python("""
        import numpy as np, photonzip
        x = np.fromfile(r'/root/workspace/PhotonZip/3rdparty/lossless/MANS/testdata/u2/exafel/exafel_59200x388_64kB.u2', dtype=np.uint16)
        try: packed = photonzip.compress(x, backend='cuda', codec_options=mans.MansOptions(warn_on_default_threads=False))
        except RuntimeError as e: print(f'SKIP:{e}'); raise SystemExit(0)
        print(int(packed.__dlpack_device__()[0]) != int(np.arange(1, dtype=np.uint8).__dlpack_device__()[0]))
    """)
    assert out == "True"


def test_host_input_can_roundtrip_back_to_cpu_via_cuda_backend():
    out = _run_python("""
        import numpy as np, torch, photonzip
        x = np.fromfile(r'/root/workspace/PhotonZip/3rdparty/lossless/MANS/testdata/u2/exafel/exafel_59200x388_64kB.u2', dtype=np.uint16)
        try: y = torch.from_dlpack(photonzip.decompress(photonzip.compress(x, backend='cuda', codec_options=mans.MansOptions(warn_on_default_threads=False)))).cpu().numpy()
        except RuntimeError as e: print(f'SKIP:{e}'); raise SystemExit(0)
        print(y.tobytes() == x.tobytes())
    """)
    assert out == "True"


def test_decompress_cuda_backend_returns_device_from_host_input():
    out = _run_python("""
        import numpy as np, torch, photonzip
        x = np.fromfile(r'/root/workspace/PhotonZip/3rdparty/lossless/MANS/testdata/u2/exafel/exafel_59200x388_64kB.u2', dtype=np.uint16)
        try: y = torch.from_dlpack(photonzip.decompress(photonzip.compress(x, backend='cuda', codec_options=mans.MansOptions(warn_on_default_threads=False))))
        except RuntimeError as e: print(f'SKIP:{e}'); raise SystemExit(0)
        print(y.device.type == 'cuda' and y.cpu().numpy().tobytes() == x.tobytes())
    """)
    assert out == "True"


def test_cuda_device_input_compresses_to_device_payload():
    out = _run_python("""
        import numpy as np, torch, photonzip
        if not torch.cuda.is_available(): print('SKIP:CUDA not available'); raise SystemExit(0)
        x = torch.from_numpy(np.fromfile(r'/root/workspace/PhotonZip/3rdparty/lossless/MANS/testdata/u2/exafel/exafel_59200x388_64kB.u2', dtype=np.uint16)).cuda()
        try: packed = photonzip.compress(x, backend='cuda', codec_options=mans.MansOptions(warn_on_default_threads=False))
        except RuntimeError as e: print(f'SKIP:{e}'); raise SystemExit(0)
        print(packed.compressed and packed.__dlpack_device__()[0] != np.arange(1, dtype=np.uint8).__dlpack_device__()[0])
    """)
    assert out == "True"


def test_cuda_device_input_and_output_stay_on_device():
    out = _run_python("""
        import numpy as np, torch, photonzip
        if not torch.cuda.is_available(): print('SKIP:CUDA not available'); raise SystemExit(0)
        x = torch.from_numpy(np.fromfile(r'/root/workspace/PhotonZip/3rdparty/lossless/MANS/testdata/u2/exafel/exafel_59200x388_64kB.u2', dtype=np.uint16)).cuda()
        try: y = torch.from_dlpack(photonzip.decompress(photonzip.compress(x, backend='cuda', codec_options=mans.MansOptions(warn_on_default_threads=False))))
        except RuntimeError as e: print(f'SKIP:{e}'); raise SystemExit(0)
        print(y.device.type == 'cuda' and torch.equal(y, x))
    """)
    assert out == "True"
