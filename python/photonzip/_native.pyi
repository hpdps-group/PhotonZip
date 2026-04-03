from typing import List
from typing import Any


class PhotonZipError(Exception): ...
class PhotonZipArray:
    codec: str
    compressed: bool
    dtype: str
    shape: List[int]
    nbytes: int
    def to_bytes(self) -> bytes: ...
    def __dlpack__(self, stream: Any = ...) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...


def list_codecs() -> List[str]: ...


def compress_tensor(
    codec_name: str,
    input: object,
    backend: str = ...,
    codec_params: List[int] = ...,
) -> PhotonZipArray: ...


def decompress_tensor(
    input: PhotonZipArray,
    backend: str = ...,
) -> PhotonZipArray: ...


def invoke_codec(
    codec_name: str,
    op_name: str,
    request: object = ...,
) -> object: ...
