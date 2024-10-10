# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for attacks."""

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from ml_collections import ConfigDict

from gcg.eval_input import EvalInput, LengthMismatchError
from gcg.model import TransformersModel
from gcg.types import BatchTokenIds
from gcg.utils import Message, SuffixManager

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Attack's output."""

    best_loss: float
    best_suffix: str
    num_queries: int
    success: bool


class BaseAttack:
    """Base class for attacks."""

    name: str = "base"  # Name of the attack
    valid_skip_modes = ("none", "seen", "visited")

    def __init__(
        self,
        config: ConfigDict,
        model,
        tokenizer: transformers.AutoTokenizer,
        suffix_manager: SuffixManager,
        not_allowed_tokens: torch.Tensor | None,
        eval_func: Any,
        **kwargs,
    ) -> None:
        """Initialize the attack."""
        _ = kwargs  # Unused
        self._num_steps: int = config.num_steps
        self._fixed_params: bool = config.fixed_params
        self._adv_suffix_init: str = config.adv_suffix_init
        self._init_suffix_len: int = config.init_suffix_len
        self._batch_size = config.batch_size
        if config.mini_batch_size <= 0:
            self._mini_batch_size = config.batch_size
        else:
            self._mini_batch_size = config.mini_batch_size
        self._log_freq: int = config.log_freq
        self._allow_non_ascii: bool = config.allow_non_ascii
        self._seed: int = config.seed
        self._seq_len: int = config.seq_len
        self._loss_temperature: float = config.loss_temperature
        self._max_queries: float = config.max_queries
        self._add_space: bool = config.add_space
        self._eval_func = eval_func

        if config.skip_mode not in self.valid_skip_modes:
            raise ValueError(
                f"Invalid skip_mode: {config.skip_mode}! Must be one of "
                f"{self.valid_skip_modes}."
            )
        self._skip_mode = config.skip_mode
        self._skip_seen = config.skip_mode == "seen"
        self._skip_visited = self._skip_seen or config.skip_mode == "visited"

        wrapped_model = TransformersModel(
            "alpaca@none",
            suffix_manager=suffix_manager,
            model=model,
            tokenizer=tokenizer,
            system_message="",
            max_tokens=100,
            temperature=0.0,
        )

        self._model = wrapped_model
        self._device = wrapped_model.device
        self._not_allowed_tokens = not_allowed_tokens.to(self._device)
        self._tokenizer = tokenizer
        self._suffix_manager = suffix_manager
        self._setup_log_file(config)

        # Runtime variables
        self._start_time = None
        self._step = None
        self._best_loss = None
        self._best_suffix = None
        self._num_queries = 0
        self._seen_suffixes = set()
        self._visited_suffixes = set()
        self._num_repeated = 0

    def _setup_log_file(self, config):
        atk_name = str(self).replace(f"{self.name}_", "")
        if config.custom_name:
            atk_name += f"_{config.custom_name}"
        log_dir = Path(config.log_dir) / self.name / atk_name
        logger.info("Logging to %s", log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{config.sample_id}.jsonl"
        # Delete log file if it exists
        log_file.unlink(missing_ok=True)
        self._log_file = log_file

    def _get_name_tokens(self) -> list[str]:
        """Create a name for this attack based on its parameters."""
        if self._init_suffix_len <= 0:
            init_suffix_len = len(self._adv_suffix_init.split())
        else:
            init_suffix_len = self._init_suffix_len
        atk_tokens = [self.name, f"len{init_suffix_len}"]
        if self._max_queries > 0:
            atk_tokens.append(f"{self._max_queries:g}query")
        else:
            atk_tokens.append(f"{self._num_steps}step")
        atk_tokens.extend(
            [
                f"bs{self._batch_size}",
                f"seed{self._seed}",
                f"l{self._seq_len}",
                f"t{self._loss_temperature}",
            ]
        )
        if self._fixed_params:
            atk_tokens.append("static")
        if self._allow_non_ascii:
            atk_tokens.append("nonascii")
        if self._skip_mode != "none":
            atk_tokens.append(self._skip_mode)
        return atk_tokens

    def __str__(self):
        return "_".join(self._get_name_tokens())

    def _sample_updates(self, optim_ids, *args, **kwargs):
        raise NotImplementedError("_sample_updates not implemented")

    def _setup_run(
        self,
        *args,
        messages: list[Message] | None = None,
        adv_suffix: str = "",
        **kwargs,
    ) -> None:
        """Set up before each attack run."""
        _ = args, kwargs  # Unused
        self._start_time = time.time()
        self._num_queries = 0
        self._step = None
        self._best_loss, self._best_suffix = float("inf"), adv_suffix
        self._seen_suffixes = set()
        self._visited_suffixes = set()
        self._num_repeated = 0
        if not self._fixed_params:
            return
        self._model.set_prefix_cache(messages)

    def _on_step_begin(self, *args, **kwargs):
        """Exectued at the beginning of each step."""

    def _save_best(self, current_loss: float, current_suffix: str) -> None:
        """Save the best loss and suffix so far."""
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._best_suffix = current_suffix

    def cleanup(self) -> None:
        """Clean up memory after run."""

    def _compute_suffix_loss(self, eval_input: EvalInput) -> torch.Tensor:
        """Compute loss given multiple suffixes.

        Args:
            eval_input: Input to evaluate. Must be EvalInput.

        Returns:
            Tuple of logits and loss.
        """
        output = self._model.compute_suffix_loss(
            eval_input,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
        )
        self._num_queries += output.num_queries
        return output.losses

    def _compute_grad(self, eval_input: EvalInput, **kwargs) -> torch.Tensor | None:
        raise NotImplementedError("_compute_grad not implemented")

    def _filter_suffixes(self, adv_suffix_ids: BatchTokenIds) -> tuple[BatchTokenIds, int]:
        """Filter out invalid adversarial suffixes."""
        skipped_suffixes = None
        if self._skip_visited:
            skipped_suffixes = self._visited_suffixes
        elif self._skip_seen:
            skipped_suffixes = self._seen_suffixes

        is_valid = self._model.filter_suffixes(
            suffix_ids=adv_suffix_ids, skipped_suffixes=skipped_suffixes
        )
        num_valid = is_valid.int().sum().item()
        adv_suffix_ids = adv_suffix_ids[is_valid]
        orig_len = adv_suffix_ids.shape[1]
        batch_size = self._batch_size

        adv_suffix_ids = adv_suffix_ids[:, :orig_len]
        if num_valid < batch_size:
            # Pad adv_suffix_ids to original batch size
            batch_pad = torch.zeros(
                (batch_size - num_valid, orig_len),
                dtype=adv_suffix_ids.dtype,
                device=adv_suffix_ids.device,
            )
            adv_suffix_ids = torch.cat([adv_suffix_ids, batch_pad], dim=0)
            logger.debug("%.3f of suffixes are invalid", 1 - num_valid / batch_size)
        else:
            # We have more valid samples than the desired batch size
            num_valid = batch_size
        adv_suffix_ids = adv_suffix_ids[:batch_size]

        if adv_suffix_ids.shape[0] == 0:
            raise RuntimeError("No valid suffixes found!")
        assert adv_suffix_ids.shape == (batch_size, orig_len)
        return adv_suffix_ids, num_valid

    def _get_next_suffix(
        self, eval_input: EvalInput, adv_suffixes: list[str], num_valid: int
    ) -> tuple[str, float]:
        """Select the suffix for the next step."""
        raise NotImplementedError("_get_next_suffix not implemented")

    @torch.no_grad()
    def run(self, messages: list[Message], target: str) -> AttackResult:
        """Run the attack."""
        if self._add_space:
            target = "▁" + target
        # Setting up init suffix
        num_failed = 0
        adv_suffix = self._adv_suffix_init
        adv_suffix_ids = self._tokenizer(
            adv_suffix, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        adv_suffix_ids.squeeze_(0)

        while True:
            if num_failed >= len(adv_suffix_ids):
                # This should never be reached because "!" x N should be valid
                raise RuntimeError("Invalid init suffix!")
            try:
                self._setup_run(messages=messages, target=target, adv_suffix=adv_suffix)
            except LengthMismatchError as e:
                logger.warning('Failing with suffix: "%s"', adv_suffix)
                logger.warning(str(e))
                logger.warning("Retrying with a new suffix...")
                # Replace the last N tokens with dummy where N is the number of
                # failed trials so far + 1
                dummy = self._tokenizer("!", add_special_tokens=False).input_ids[0]
                adv_suffix_ids[-num_failed - 1 :] = dummy
                adv_suffix = self._tokenizer.decode(
                    adv_suffix_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                num_failed += 1
                continue
            break
        num_fixed_tokens = self._model.num_fixed_tokens

        logger.debug("Starting attack with suffix: %s", adv_suffix)
        assert adv_suffix_ids.ndim == 1, adv_suffix_ids.shape
        #logger.debug(
            #"\nInitialized suffix (%d tokens):\n%s",
            #len(adv_suffix_ids),
            #adv_suffix,
        #)

        # =============== Prepare inputs and determine slices ================ #
        eval_input = self._suffix_manager.gen_eval_inputs(
            messages,
            adv_suffix,
            target,
            num_fixed_tokens=num_fixed_tokens,
            max_target_len=self._seq_len,
        )
        eval_input.to("cuda")
        optim_slice = eval_input.optim_slice
        passed = True

        for i in range(self._num_steps):
            self._step = i
            self._on_step_begin()

            dynamic_input_ids = self._suffix_manager.get_input_ids(messages, adv_suffix, target)[0][
                num_fixed_tokens:
            ]
            dynamic_input_ids = dynamic_input_ids.to("cuda")
            optim_ids = dynamic_input_ids[optim_slice]
            eval_input.dynamic_input_ids = dynamic_input_ids
            eval_input.suffix_ids = optim_ids

            # Compute grad as needed (None if no-grad attack)
            # pylint: disable=assignment-from-none
            # computes for entire batch
            token_grads = self._compute_grad(eval_input)
            adv_suffix_ids = self._sample_updates(
                optim_ids=optim_ids,
                grad=token_grads,
                optim_slice=optim_slice,
            )

            # Filter out "invalid" adversarial suffixes
            adv_suffix_ids, num_valid = self._filter_suffixes(adv_suffix_ids)
            adv_suffixes = self._tokenizer.batch_decode(
                adv_suffix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            self._seen_suffixes.update(adv_suffixes)
            eval_input.suffix_ids = adv_suffix_ids

            # Compute loss on model
            # computes only minibatch loss and logits (Which may be a problem?)
            losses = self._compute_suffix_loss(eval_input)
            idx = losses[:num_valid].argmin()
            adv_suffix = adv_suffixes[idx]
            loss = losses[idx].item()
            current_loss = losses[idx].item()

            # Save the best candidate and update visited suffixes
            self._save_best(current_loss, adv_suffix)
            self._visited_suffixes.add(adv_suffix)

            if (i+1) % self._log_freq == 0 or i == 0:
                # Logging
                self._num_queries += 1
                result = self._eval_func(adv_suffix, messages)
                passed = result[1] == 0
                self.log(
                    log_dict={
                        "loss": current_loss,
                        "best_loss": self._best_loss,
                        "success_begin_with": result[1] == 1,
                        "success_in_response": result[0] == 1,
                        "suffix": adv_suffix,
                        "generated": result[2][0][0],  # last message
                        #"num_cands": adv_suffix_ids.shape[0], 
                    },
                )
            del token_grads, dynamic_input_ids
            gc.collect()

            if not passed:
                logger.info("Attack succeeded! Early stopping...")
                self._best_suffix = adv_suffix
                break
            if self._num_queries >= self._max_queries > 0:
                logger.info("Max queries reached! Finishing up...")
                break

        # Evaluate last suffix on target model (this step is redundant on
        # attacks that do not use proxy model).
        eval_input.suffix_ids = self._tokenizer(
            adv_suffix, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        loss = self._model.compute_suffix_loss(eval_input, batch_size=self._mini_batch_size).losses
        self._save_best(loss.min().item(), adv_suffix)

        attack_result = AttackResult(
            best_loss=self._best_loss,
            best_suffix=self._best_suffix,
            num_queries=self._num_queries,
            success=not passed,
        )
        self._step += 1
        return attack_result

    def format(self, d, tab=0):
        s = ['{\n']
        for k,v in d.items():
            if isinstance(v, dict): v = format(v, tab+1)
            else: v = repr(v)
            s.append('%s%r: %s,\n' % ('  '*tab, k, v))
        s.append('%s}' % ('  '*tab))
        return ''.join(s)

    def log(self, step: int | None = None, log_dict: dict[str, Any] | None = None) -> None:
        """Log data using logger from a single step."""
        step = step or self._step
        log_dict["mem"] = torch.cuda.max_memory_allocated() / 1e9
        log_dict["time_per_step_s"] = (time.time() - self._start_time) / (step + 1)
        log_dict["queries"] = self._num_queries
        log_dict["time_min"] = (time.time() - self._start_time) / 60
        log_dict['sample_id'] = str(self._log_file).split('/')[-1].split('.')[0]
        message = ""
        for key, value in log_dict.items():
            if "loss" in key:
                try:
                    value = f"{value:.4f}"
                except TypeError:
                    pass
            elif key == "mem":
                value = f"{value:.2f}GB"
            elif key == "time_per_step":
                value = f"{value:.2f}s"
            message += f"{key}={value}, "
        logger.info("[step: %4d/%4d] %s", step, self._num_steps, self.format(log_dict, 2))
        log_dict["step"] = step

        # Convert all tensor values to lists or floats
        def tensor_to_serializable(val):
            if isinstance(val, torch.Tensor):
                return val.tolist() if val.numel() > 1 else val.item()
            return val

        log_dict = {k: tensor_to_serializable(v) for k, v in log_dict.items()}
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")
