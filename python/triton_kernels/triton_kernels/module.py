import inspect
from typing import Any

import torch
from torch.autograd.function import _is_setup_context_defined

from .tensor import Tensor
from .tensor_types import TypeAssertion


class Module(torch.autograd.Function):
    TYPES: dict[str, TypeAssertion] = {}

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        if not _is_setup_context_defined(cls.setup_context):
            # make room for context, pass as first argument
            forward_args = (None,) + args
        else:
            forward_args = args
        sig = inspect.signature(cls.forward)
        bound_args = sig.bind(*forward_args, **kwargs)
        bound_args.apply_defaults()

        for name, value in bound_args.arguments.items():
            is_tensor = isinstance(value, Tensor)
            type_assertion = cls.TYPES.get(name, None)
            if is_tensor:
                if type_assertion is None:
                    raise ValueError(
                        f"{cls}: parameter {name} is a tensor, but has no type assertion"
                    )
                if not type_assertion.is_valid(value):
                    raise ValueError(
                        f"{cls}: parameter {name}: type assertion failed: {type_assertion}"
                    )
            elif type_assertion is not None:
                # Allow (unchecked) torch.Tensors
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"{cls}: parameter {name} has type assertion, but is not a tensor (it is {type(value)})"
                    )

        return super().apply(*args, **kwargs)
