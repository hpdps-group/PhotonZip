from importlib import import_module


try:
    _native = import_module("photonzip._native")
except ModuleNotFoundError:
    _native = import_module("_native")


def list_codecs():
    return _native.list_codecs()


def invoke_codec(codec, operation, request=None):
    if request is None:
        request = {}
    return _native.invoke_codec(codec, operation, request=request)


def compress(
    tensor,
    codec=None,
    backend="auto",
    codec_params=None,
    codec_options=None,
):
    if codec_params is not None and codec_options is not None:
        raise TypeError("Pass either codec_params or codec_options, not both.")

    if codec_options is not None:
        if not hasattr(codec_options, "to_codec_params") or not hasattr(codec_options, "codec"):
            raise TypeError("codec_options must provide codec and to_codec_params().")
        if codec is not None and codec != codec_options.codec:
            raise ValueError(
                f"codec={codec!r} does not match codec_options.codec={codec_options.codec!r}."
            )
        codec = codec_options.codec
        codec_params = codec_options.to_codec_params(tensor=tensor)

    if codec is None:
        codec = "mans"

    if codec_params is None:
        codec_params = []
    return _native.compress_tensor(codec, tensor, backend=backend, codec_params=codec_params)


def decompress(data, **kwargs):
    import torch

    return torch.from_dlpack(_native.decompress_tensor(data, **kwargs))


compress_tensor = compress
decompress_tensor = decompress
