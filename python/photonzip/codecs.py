from __future__ import annotations

from abc import ABC, abstractmethod


class Codec(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compress(
        self,
        tensor,
        *,
        backend="auto",
        codec_params=None,
        codec_options=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def decompress(self, data, *, backend="auto", **kwargs):
        raise NotImplementedError

    def invoke(self, operation, request=None):
        raise NotImplementedError(f"Codec {self.name!r} does not implement operation {operation!r}.")


from .codec import discover_codecs
from .codec_registry import get_codec_handler, list_registered_codecs


discover_codecs()


def list_codecs():
    return list_registered_codecs()


def invoke_codec(codec, operation, request=None):
    handler = get_codec_handler(codec)
    if handler is None:
        raise ValueError(f"Unknown codec: {codec!r}.")
    return handler.invoke(operation, request=request)


def compress(
    tensor,
    codec="mans",
    backend="auto",
    codec_params=None,
    codec_options=None,
):
    if codec_params is not None and codec_options is not None:
        raise TypeError("Pass either codec_params or codec_options, not both.")

    if codec_options is not None:
        if not hasattr(codec_options, "codec"):
            raise TypeError("codec_options must provide codec.")
        if codec is not None and codec != codec_options.codec:
            raise ValueError(
                f"codec={codec!r} does not match codec_options.codec={codec_options.codec!r}."
            )
        codec = codec_options.codec


    handler = get_codec_handler(codec)
    if handler is None:
        raise ValueError(f"Unknown codec: {codec!r}.")
    return handler.compress(
        tensor,
        backend=backend,
        codec_params=codec_params,
        codec_options=codec_options,
    )


def decompress(data, backend="auto", codec=None, **kwargs):
    codec_name = codec if codec is not None else getattr(data, "codec", None)
    if codec_name is None:
        raise ValueError("Unable to determine codec for decompression. Pass codec explicitly.")
    handler = get_codec_handler(codec_name)
    if handler is None:
        raise ValueError(f"Unknown codec: {codec_name!r}.")
    return handler.decompress(data, backend=backend, **kwargs)


compress_tensor = compress
decompress_tensor = decompress
