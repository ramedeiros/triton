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
    def full_map(self) -> Sequence[ShardLocation]:
        """Return the entire mapping."""
        ...

    @abstractmethod
    def map(self, idxs: torch.Tensor) -> torch.Tensor:
        """Return the mapping of a set of indices.

        Given a set of indices (of shape [k]), return a tensor of shape [replication_factor, k, 2]
        locations. ret[.., 0] is the rank; ret[..., 1] is the local index on that rank.
        """
        ...

    @property
    def is_fully_replicated(self) -> bool:
        return False


@dataclass(frozen=True, kw_only=True)
class LocalSharding(Sharding):
    def full_map(self, size: int) -> Sequence[ShardLocation]:
        return (ShardLocation(shard=slice(0, size)),)

    def map(self, idxs: torch.Tensor, size: int) -> torch.Tensor:
        my_rank = self.process_group.rank
        ranks = torch.full_like(idxs, my_rank)
        return torch.stack((ranks, idxs), dim=1).unsqueeze(0)


@dataclass(frozen=True, kw_only=True)
class RangeSharding(Sharding):
    replication_factor: int = 1

    @property
    def n_shards(self) -> int:
        return self.process_group.world_size // self.replication_factor

    @property
    def is_fully_replicated(self):
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


if __name__ == "__main__":
    sharding = RangeSharding(n_ranks=8, replication_factor=2)
    idxs = torch.arange(18)
    print(sharding.get_full(size=18))
    print(sharding.map(idxs, size=18))
