# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GCG Attack."""

import numpy as np
import torch
from ml_collections import ConfigDict

from gcg.base import BaseAttack
from gcg.eval_input import EvalInput
from gcg.types import BatchTokenIds


def _rand_permute(size, device: str = "cuda", dim: int = -1):
    return torch.argsort(torch.rand(size, device=device), dim=dim)


class GCGAttack(BaseAttack):
    """GCG Attack (see https://llm-attacks.org/)."""

    name: str = "gcg"

    def __init__(self, config: ConfigDict, *args, **kwargs) -> None:
        """Initialize GCG attack."""
        self._topk = config.topk
        self._num_coords: tuple[int, int] = config.num_coords
        self._mu: float = config.mu
        if not isinstance(self._num_coords, tuple) or len(self._num_coords) != 2:
            raise ValueError(f"num_coords must be tuple of two ints, got {self._num_coords}")

        # Init base class after setting parameters because it will call
        # _get_name_tokens() which uses the parameters. See below.
        super().__init__(config, *args, **kwargs)
        self._momentum: torch.Tensor | None = None

    def _get_name_tokens(self) -> list[str]:
        atk_tokens = super()._get_name_tokens()
        atk_tokens.append(f"k{self._topk}")
        if any(c != 1 for c in self._num_coords):
            if self._num_coords[0] == self._num_coords[1]:
                atk_tokens.append(f"c{self._num_coords[0]}")
            else:
                atk_tokens.append(f"c{self._num_coords[0]}-{self._num_coords[1]}")
        if self._mu != 0:
            atk_tokens.append(f"m{self._mu}")
        return atk_tokens

    def _param_schedule(self):
        num_coords = round(
            self._num_coords[0]
            + (self._num_coords[1] - self._num_coords[0]) * self._step / self._num_steps
        )
        return num_coords

    @torch.no_grad()
    def _compute_grad(self, eval_input: EvalInput, **kwargs) -> torch.Tensor:
        _ = kwargs  # unused
        grad = self._model.compute_grad(
            eval_input,
            temperature=self._loss_temperature,
            return_logits=True,
            **kwargs,
        )
        if self._mu == 0:
            return grad

        # Calculate momentum term
        if self._momentum is None:
            self._momentum = torch.zeros_like(grad)
        self._momentum.mul_(self._mu).add_(grad)
        return self._momentum

    @torch.no_grad()
    def _sample_updates(
        self,
        optim_ids,
        *args,
        grad: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchTokenIds:
        _ = args, kwargs  # unused
        assert isinstance(grad, torch.Tensor), "grad is required for GCG!"
        assert len(grad) == len(optim_ids), (
            f"grad and optim_ids must have the same length ({len(grad)} vs " f"{len(optim_ids)})!"
        )
        device = grad.device
        num_coords = self._param_schedule()
        num_coords = min(num_coords, len(optim_ids))
        if self._not_allowed_tokens is not None:
            grad[:, self._not_allowed_tokens.to(device)] = np.infty

        # pylint: disable=invalid-unary-operand-type
        top_indices = (-grad).topk(self._topk, dim=1).indices

        batch_size = int(self._batch_size * 1.25)
        old_token_ids = optim_ids.repeat(batch_size, 1)

        if num_coords == 1:
            # Each position will have `batch_size / len(optim_ids)` candidates
            new_token_pos = torch.arange(
                0,
                len(optim_ids),
                len(optim_ids) / batch_size,
                device=device,
            ).type(torch.int64)
            # Get random indices to select from topk
            # rand_idx: [seq_len, topk, 1]
            rand_idx = _rand_permute((len(optim_ids), self._topk, 1), device=device, dim=1)
            # Get the first (roughly) batch_size / seq_len indices at each position
            rand_idx = torch.cat(
                [r[: (new_token_pos == i).sum()] for i, r in enumerate(rand_idx)],
                dim=0,
            )
            assert rand_idx.shape == (batch_size, 1), rand_idx.shape
            new_token_val = torch.gather(top_indices[new_token_pos], 1, rand_idx)
            new_token_ids = old_token_ids.scatter(1, new_token_pos.unsqueeze(-1), new_token_val)
        else:
            # Random choose positions to update
            new_token_pos = _rand_permute((batch_size, len(optim_ids)), device=device, dim=1)[
                :, :num_coords
            ]
            # Get random indices to select from topk
            rand_idx = torch.randint(0, self._topk, (batch_size, num_coords, 1), device=device)
            new_token_val = torch.gather(top_indices[new_token_pos], -1, rand_idx)
            new_token_ids = old_token_ids
            for i in range(num_coords):
                new_token_ids.scatter_(1, new_token_pos[:, i].unsqueeze(-1), new_token_val[:, i])

        assert new_token_ids.shape == (
            batch_size,
            len(optim_ids),
        ), new_token_ids.shape
        return new_token_ids

    def _get_next_suffix(
        self, eval_input: EvalInput, adv_suffixes: list[str], num_valid: int
    ) -> tuple[str, float]:
        """Select the suffix for the next step."""
        # Compute loss on model
        output = self._model.compute_suffix_loss(
            eval_input,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
        )
        losses = output.losses
        self._num_queries += output.num_queries

        idx = losses[:num_valid].argmin()
        adv_suffix = adv_suffixes[idx]
        loss = losses[idx].item()
        return adv_suffix, loss
