from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .placement import ProcessGroup, default_process_group


@dataclass(frozen=True, kw_only=True, slots=True)
class ShardLocation:
    rank: int
    shard: slice


@dataclass(frozen=True, kw_only=True)
class Sharding(ABC):
    process_group: ProcessGroup = field(default_factory=default_process_group)

    @abstractmethod
    def full_map(self, size: int) -> Sequence[ShardLocation]:
        """Return the entire mapping."""
        ...

    @abstractmethod
    def map(self, idxs: torch.Tensor, size: int) -> torch.Tensor:
        """Return the mapping of a set of indices.

        Given a set of indices (of shape [k]), return a tensor of shape [replication_factor, k, 2]
        locations. ret[.., 0] is the rank; ret[..., 1] is the local index on that rank.
        """
        ...

    @abstractmethod
    def range_for_rank(self, rank: int, size: int) -> slice:
        """Return the contiguous index range mapped to a given rank."""
        ...

    @property
    def is_fully_replicated(self) -> bool:
        return False

    def uniform_width(self, size: int) -> int | None:
        return None


@dataclass(frozen=True, kw_only=True)
class LocalSharding(Sharding):
    def full_map(self, size: int) -> Sequence[ShardLocation]:
        return (ShardLocation(shard=slice(0, size)),)

    def map(self, idxs: torch.Tensor, size: int) -> torch.Tensor:
        my_rank = self.process_group.rank
        ranks = torch.full_like(idxs, my_rank)
        return torch.stack((ranks, idxs), dim=1).unsqueeze(0)

    def range_for_rank(self, rank: int, size: int) -> slice:
        return slice(0, size if rank == self.process_group.rank else 0)

    def uniform_width(self, size: int) -> int | None:
        return size if self.process_group.world_size == 1 else None


@dataclass(frozen=True, kw_only=True)
class RangeSharding(Sharding):
    replication_factor: int = 1

    @property
    def n_shards(self) -> int:
        return self.process_group.world_size // self.replication_factor

    @property
    def is_fully_replicated(self) -> bool:
        return self.n_shards == 1

    # n_shards = n_ranks / replication_factor
    # Shard j is mapped to ranks [j * replication_factor, (j + 1) * replication_factor)
    # Index i is mapped to shard floor(i * n_shards / size)
    #
    # if q, r = divmod(size, n_shards)
    # then shards [0, r) will have size q + 1
    #      shards [r+1, s) will have size q

    def __post_init__(self):
        assert self.process_group.world_size % self.replication_factor == 0

    def full_map(self, size: int) -> Sequence[ShardLocation]:
        q, r = divmod(size, self.n_shards)
        locations = []
        start = 0
        for shard in range(self.n_shards):
            shard_size = q + (shard < r)
            end = start + shard_size
            for replica in range(self.replication_factor):
                locations.append(
                    ShardLocation(
                        rank=shard * self.replication_factor + replica, shard=slice(start, end)
                    )
                )
            start = end
        return locations

    def map(self, idxs: torch.Tensor, size: int) -> Sequence[torch.Tensor]:
        r = self.replication_factor
        n = idxs.shape[0]
        idxs = torch.atleast_1d(idxs)
        ranks = idxs * self.n_shards // size * self.replication_factor

        idxs = idxs.unsqueeze(0).expand(r, n)
        ranks = ranks.unsqueeze(0).expand(r, n)
        replication = (
            torch.arange(r, dtype=idxs.dtype, device=idxs.device).unsqueeze(1).expand(r, n)
        )

        return torch.stack((ranks + replication, idxs), dim=2)

    def range_for_rank(self, rank: int, size: int) -> slice:
        shard = rank // self.replication_factor
        q, r = divmod(size, self.n_shards)
        start = shard * q + min(shard, r)
        end = start + q + (shard < r)
        return slice(start, end)

    def uniform_width(self, size: int) -> int | None:
        q, r = divmod(size, self.n_shards)
        return q if r == 0 else None


if __name__ == "__main__":
    SIZE_MAX = 18
    sharding = RangeSharding(replication_factor=2)
    idxs = torch.arange(SIZE_MAX)
    print(sharding.full_map(SIZE_MAX))
    print(sharding.map(idxs, SIZE_MAX))
