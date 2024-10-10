# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from gcg.types import BatchTokenIds, BatchTokenProbs, TokenIds, TokenProbs

SuffixIds = BatchTokenIds | TokenIds
TargetIds = BatchTokenIds | TokenIds | BatchTokenProbs | TokenProbs


class LengthMismatchError(Exception):
    """Length of token ids does not match the corresponding slice."""


@dataclass
class EvalInput:
    """All parameters needed to compute outputs and loss."""

    dynamic_input_ids: TokenIds | None = None
    optim_slice: slice | None = None
    target_slice: slice | None = None
    loss_slice: slice | None = None
    suffix_ids: SuffixIds | None = None
    target_ids: TargetIds | None = None
    

    def __post_init__(self):
        self.check_props()

    def check_props(self):
        """Check that all properties are valid."""
        self._check_suffix_ids(self.suffix_ids, self.optim_slice)
        self._check_target_ids(self.target_ids, self.target_slice)
        self._check_input_ids(
            self.dynamic_input_ids,
            self.optim_slice,
            self.target_slice,
            self.loss_slice,
        )

    def __setattr__(self, prop, val):
        if prop == "suffix_ids":
            self._check_suffix_ids(val, self.optim_slice)
        elif prop == "target_ids":
            self._check_target_ids(val, self.target_slice)
        elif prop == "dynamic_input_ids":
            self._check_input_ids(val, self.optim_slice, self.target_slice, self.loss_slice)
        super().__setattr__(prop, val)

    @staticmethod
    def _check_input_ids(
        input_ids, optim_slice: slice, target_slice: slice, loss_slice: slice
    ) -> None:
        if input_ids.ndim != 1:
            raise ValueError(f"dynamic_input_ids must be 1D tensor! Got {input_ids.shape}.")
        inpt_len = len(input_ids)
        if any(
            inpt_len < slce.stop
            for slce in (optim_slice, target_slice, loss_slice)
            if slce is not None
        ):
            raise LengthMismatchError(
                f"Length of dynamic_input_ids ({inpt_len}) is shorter than "
                f"optim_slice ({optim_slice.stop}), target_slice "
                f"({target_slice.stop}), or loss_slice ({loss_slice.stop})!"
            )

    @staticmethod
    def _check_suffix_ids(suffix_ids, optim_slice: slice) -> None:
        """Check that suffix_ids is valid."""
        if suffix_ids is None or optim_slice is None:
            return
        assert suffix_ids.ndim in (1, 2)
        suffix_len = suffix_ids.shape[-1]
        num_optim_tokens = optim_slice.stop - optim_slice.start
        #print(suffix_len, num_optim_tokens)
        if suffix_len != num_optim_tokens:
            raise LengthMismatchError(
                f"Length of given suffix_ids ({suffix_len}) does not match "
                f"optim_slice ({num_optim_tokens})!\nsuffix_ids: {suffix_ids}\n"
                f"optim_slice: {optim_slice}"
            )

    @staticmethod
    def _check_target_ids(target_ids, target_slice: slice) -> None:
        """Check that target_ids is valid."""
        if target_ids is None or target_slice is None:
            return
        if target_ids.dtype == torch.long:
            assert target_ids.ndim in (1, 2)
            target_len = target_ids.shape[-1]
        else:
            assert target_ids.ndim in (2, 3)
            target_len = target_ids.shape[-2]
        num_target_tokens = target_slice.stop - target_slice.start
        if target_len != num_target_tokens:
            raise LengthMismatchError(
                f"Length of given target_ids ({target_ids}) does not match "
                f"target_slice ({num_target_tokens})!\ntarget_ids: {target_ids}"
                f"\ntarget_slice: {target_slice}"
            )

    def to(self, device: str | torch.device) -> None:
        """Move all tensors to the given device."""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device, non_blocking=True))

    def print(self) -> str:
        """Return human-readable string representation of this object."""
        string = "[EvalInput]:\n"
        string += (
            f"  dynamic_input_ids {tuple(self.dynamic_input_ids.shape)}:\n"
            f"{self.dynamic_input_ids}\n"
        )
        string += f"  suffix_ids {tuple(self.suffix_ids.shape)}:\n"
        if self.suffix_ids.ndim == 1:
            string += f"{self.suffix_ids}\n"
        else:
            string += f"{self.suffix_ids[0]}...\n"
        string += f"  target_ids {tuple(self.target_ids.shape)}:\n"
        if self.target_ids.ndim == 1:
            string += f"{self.target_ids}\n"
        else:
            string += f"{self.target_ids[0]}...\n"
        string += f"  optim_slice: {self.optim_slice}\n"
        string += f"  target_slice: {self.target_slice}\n"
        string += f"  loss_slice: {self.loss_slice}"
        return string


@dataclass
class BatchEvalInput:
    """Batch of EvalInput objects."""

    batch_eval_input: list[EvalInput]


def merge_eval_inputs(src: EvalInput, tgt: EvalInput | None) -> EvalInput:
    """Merge two EvalInput objects and return a new one.

    Copy all attributes from `src` to `tgt`, except for those that are not None
    in `tgt`.

    Args:
        src: Source eval input.
        tgt: Target eval input.

    Returns:
        New eval input object.
    """
    tgt = tgt or EvalInput()
    new_eval_input = EvalInput()
    for k, v in tgt.__dict__.items():
        if v is None:
            setattr(new_eval_input, k, getattr(src, k))
        else:
            setattr(new_eval_input, k, v)
    return new_eval_input
