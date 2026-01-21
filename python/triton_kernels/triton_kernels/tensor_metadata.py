from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Sequence

import torch

from .tensor_details.dtype import FloatType, IntegerType
from .tensor_details.sharding import LocalSharding, Sharding, ShardLocation


@dataclass(kw_only=True)
class TensorSharding:
    dim: int
    sharding: Sharding
    sharding_size: int
    max_dim_size: int | None = None

    def full_map(self) -> Sequence[ShardLocation]:
        return self.sharding.full_map(self.sharding_size)

    def map(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.sharding.map(idxs, self.sharding_size)

    def range_for_rank(self, rank: int, clamp: bool = True) -> slice:
        s = self.sharding.range_for_rank(rank, self.sharding_size)
        assert s.step in (1, None)
        if clamp and self.max_dim_size is not None:
            return (
                slice(0, 0)
                if s.start >= self.max_dim_size
                else slice(s.start, min(s.stop, self.max_dim_size))
            )
        return s

    @property
    def uniform_width(self) -> int | None:
        return self.sharding.uniform_width(self.sharding_size)

    @property
    def is_fully_replicated(self) -> bool:
        return self.sharding.is_fully_replicated

    @property
    def is_local(self) -> bool:
        return self.sharding.is_local


@dataclass(kw_only=True)
class TensorMetadata:
    dtype: IntegerType | FloatType
    shape: list[int | torch.Tensor] | None
    # Maximum size in each dimension. Note: shape_max[i] >= shape[i] if the shape is known
    # statically on dimension i.
    shape_max: list[int | None] | None = None

    sharding: InitVar[Sharding | None] = None
    sharding_dim: InitVar[int | None] = None

    # Dimension indices whose sizes are not known at compile time.
    dynamic_dims: list[int] = field(init=False)
    tensor_sharding: TensorSharding = field(init=False, default=None)

    def __post_init__(
        self,
        sharding: Sharding | None = None,
        sharding_dim: int | None = None,
        *,
        default_shape: list[int] | None = None,
    ) -> None:
        if self.shape is None:
            if default_shape is None:
                raise ValueError("shape must be provided")
            if self.dtype.bitwidth < 8:
                raise ValueError("shape must be provided for sub-byte types")
            if self.sharding is not None:
                raise ValueError("shape must be provided if sharding")
            self.shape = default_shape

        if (sharding is not None) != (sharding_dim is not None):
            raise ValueError("sharding and sharding_dim must be specified together")

        self.shape = list(self.shape)

        # validate shape: all elements must be `int` or numel-1 `torch.Tensor`
        is_int = lambda s: isinstance(s, int)
        is_item = lambda s: hasattr(s, "numel") and s.numel() == 1
        assert all(is_int(s) or is_item(s) for s in self.shape)

        # initialize shape_max
        if self.shape_max is None:
            self.shape_max = [None] * len(self.shape)
        elif len(self.shape) != len(self.shape_max):
            raise ValueError(
                f"Mismatched shape ({len(self.shape)}) / shape_max ({len(self.shape_max)}) lengths"
            )

        self.dynamic_dims = []
        for i, (s, smax) in enumerate(zip(self.shape, self.shape_max)):
            if is_int(s):
                if smax is None:
                    self.shape_max[i] = s
                elif s > smax:
                    raise ValueError(f"shape[{i}]={s} > shape_max[{i}]={smax}")
            else:
                if smax is None:
                    raise ValueError(f"shape_max[{i}] may only be None for static shapes")
                self.dynamic_dims.append(i)

        if sharding is not None:
            assert not sharding.is_local, (
                "Do not use LocalSharding directly; use get_sharding_local_if_unset() if needed"
            )
            if sharding_dim < 0 or sharding_dim >= len(self.shape):
                raise ValueError("sharding_dim out of range")
            s = self.shape[sharding_dim]
            smax = self.shape_max[sharding_dim]
            self.tensor_sharding = TensorSharding(
                dim=sharding_dim,
                sharding=sharding,
                sharding_size=smax,
                max_dim_size=s if is_int(s) and s < smax else None,
            )


def get_sharding_local_if_unset(t: TensorMetadata, dim: int) -> TensorSharding:
    if ts := t.tensor_sharding:
        if dim != ts.dim:
            raise ValueError(f"Tensor is already sharded along dimension {ts.dim}")
        return ts

    s = t.shape[dim]
    smax = t.shape_max[dim]
    return TensorSharding(
        dim=dim,
        sharding=LocalSharding(),
        sharding_size=smax,
        max_dim_size=s if isinstance(s, int) and s < smax else None,
    )
