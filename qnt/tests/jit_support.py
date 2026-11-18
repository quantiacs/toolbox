from typing import Protocol, cast

from qnt.tests.hl_support import Floats


class PlainFunction(Protocol):
    def __call__(self, *args: object) -> Floats: ...


class JitFunction(PlainFunction, Protocol):
    @property
    def py_func(self) -> PlainFunction: ...


def as_jit(dispatcher: object) -> JitFunction:
    return cast("JitFunction", dispatcher)
