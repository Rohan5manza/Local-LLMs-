from contextlib import ContextDecorator
from typing import (
    Any,
    Iterable,
    Literal,
    overload,
)

def get_option(pat: str) -> Any: ...
def set_option(pat: str, val: object) -> None: ...
def reset_option(pat: str) -> None: ...
@overload
def describe_option(pat: str, _print_desc: Literal[False]) -> str: ...
@overload
def describe_option(pat: str, _print_desc: Literal[True] = ...) -> None: ...

class DictWrapper:
    def __init__(self, d: dict[str, Any], prefix: str = ...) -> None: ...
    def __setattr__(
        self, key: str, val: str | bool | int | DictWrapper | None
    ) -> None: ...
    def __getattr__(self, key: str) -> str | bool | int | DictWrapper | None: ...
    def __dir__(self) -> Iterable[str]: ...

options: DictWrapper = ...

class option_context(ContextDecorator):
    def __init__(self, /, pat: str, val: Any, *args: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> None: ...

class OptionError(AttributeError, KeyError): ...