from abc import ABC, abstractmethod


class CodecOptions(ABC):
    @property
    @abstractmethod
    def codec(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_codec_params(self, *, tensor=None) -> list[int]:
        raise NotImplementedError


class CodecOperation(ABC):
    @property
    @abstractmethod
    def codec(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def operation(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_request(self) -> dict:
        raise NotImplementedError
