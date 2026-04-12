from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .codecs import Codec


_REGISTRY: dict[str, "Codec"] = {}


def register_codec(codec: "Codec") -> "Codec":
    _REGISTRY[codec.name] = codec
    return codec


def get_codec_handler(name: str) -> "Codec" | None:
    return _REGISTRY.get(name)


def list_registered_codecs() -> list[str]:
    return sorted(_REGISTRY)
