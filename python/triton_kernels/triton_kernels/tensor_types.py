from __future__ import annotations

from dataclasses import KW_ONLY, InitVar, dataclass, field
from types import NotImplementedType
from typing import Sequence, overload

from .tensor_metadata import TensorMetadata


@dataclass
class TypeAssertion:
    def is_valid(self, t: TensorMetadata) -> bool:
        return False

    @overload
    def __and__(self, other: TypeAssertion) -> TypeAssertion: ...

    @overload
    def __and__(self, other: object) -> NotImplementedType: ...

    def __and__(self, other: object) -> TypeAssertion | NotImplementedType:
        if not isinstance(other, TypeAssertion):
            return NotImplemented

        return And(left=self, right=other)

    @overload
    def __or__(self, other: TypeAssertion) -> TypeAssertion: ...

    @overload
    def __or__(self, other: object) -> NotImplementedType: ...

    def __or__(self, other: object) -> TypeAssertion | NotImplementedType:
        if not isinstance(other, TypeAssertion):
            return NotImplemented

        return Or(left=self, right=other)

    def __invert__(self) -> TypeAssertion:
        return Not(self)


@dataclass
class Unary(TypeAssertion):
    item: TypeAssertion


@dataclass
class Not(Unary):
    def is_valid(self, t: TensorMetadata) -> bool:
        return not self.item.is_valid(t)


@dataclass
class Binary(TypeAssertion):
    left: TypeAssertion
    right: TypeAssertion


@dataclass
class And(Binary):
    def is_valid(self, t: TensorMetadata) -> bool:
        return self.left.is_valid(t) and self.right.is_valid(t)


@dataclass
class Or(Binary):
    def is_valid(self, t: TensorMetadata) -> bool:
        return self.left.is_valid(t) or self.right.is_valid(t)


@dataclass
class Atom(TypeAssertion):
    pass


@dataclass
class Unsharded(Atom):
    def is_valid(self, t: TensorMetadata) -> bool:
        return t.tensor_sharding is None


@dataclass
class Sharded(Atom):
    dim: int
    _: KW_ONLY
    uniform: bool = False

    def is_valid(self, t: TensorMetadata) -> bool:
        s = t.tensor_sharding
        return (
            s is not None and self.dim == s.dim and not (self.uniform and s.uniform_width is None)
        )
