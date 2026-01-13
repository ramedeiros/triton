from dataclasses import dataclass
from functools import cache
from typing import Protocol

import torch.distributed as tdist


class ProcessGroup(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def get_global_rank(self, group_rank: int) -> int: ...

    def get_group_rank(self, global_rank: int) -> int: ...


@dataclass(frozen=True, kw_only=True)
class _TorchProcessGroup:
    group: tdist.ProcessGroup | None = None

    @property
    def rank(self) -> int:
        return tdist.get_rank(self.group)

    @property
    def world_size(self) -> int:
        return tdist.get_world_size(self.group)

    def get_global_rank(self, group_rank: int) -> int:
        return group_rank if self.group is None else tdist.get_global_rank(self.group, group_rank)

    def get_group_rank(self, global_rank: int) -> int:
        return global_rank if self.group is None else tdist.get_group_rank(self.group, global_rank)


@cache
def default_process_group() -> ProcessGroup:
    return _TorchProcessGroup()


def torch_process_group(group: tdist.ProcessGroup) -> ProcessGroup:
    return _TorchProcessGroup(group=group)
