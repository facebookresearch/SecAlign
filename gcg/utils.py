# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import fastchat
import torch
import transformers
from fastchat.conversation import get_conv_template
from transformers import AutoModelForCausalLM, AutoTokenizer

from gcg.eval_input import EvalInput
from gcg.types import PrefixCache

logger = logging.getLogger(__name__)


class Role(Enum):
    USER = 1
    ASSISTANT = 2
    SYSTEM = 3


@dataclass
class Message:
    role: Role
    content: str

    def __str__(self):
        return f"[{self.role.name.title()}]: {self.content}"

    @staticmethod
    def serialize(messages, user_only=False):
        if not isinstance(messages, list):
            messages = [messages]
        if user_only:
            messages = [
                {"role": m.role.name, "content": m.content} for m in messages if m.role == Role.USER
            ]
        else:
            messages = [{"role": m.role.name, "content": m.content} for m in messages]
        return messages

    @staticmethod
    def unserialize(messages: Union[dict, List[dict]]):
        if not isinstance(messages, list):
            messages = [messages]
        messages = [Message(Role[m["role"]], m["content"]) for m in messages]
        return messages


def load_model_and_tokenizer(
    model_name: str,
    tokenizer_path=None,
    device="cuda:0",
    load_in_8bit=None,
    use_system_instructions: bool = False,
    system_message: str | None = None,
    max_tokens: int = 512,
    temperature: float = 1.0,
    **kwargs,
):
    from gcg.model import TransformersModel

    template_name, model_path = model_name.split("@")
    model_path = os.path.expanduser(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_8bit=load_in_8bit,
        **kwargs,
    )
    if not load_in_8bit or load_in_8bit is None:
        model = model.to(device, non_blocking=True)
    model = model.eval()

    tokenizer_path = tokenizer_path or model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        use_system_instructions=use_system_instructions,
        conv_template=fastchat.conversation.get_conv_template(template_name),
    )

    # TODO(robust): Don't hard code max_tokens in case target is long
    # Longest target in Toxicity scenario is ~40 tokens
    wrapped_model = TransformersModel(
        model_name,
        suffix_manager=suffix_manager,
        model=model,
        tokenizer=tokenizer,
        system_message=system_message,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return wrapped_model, tokenizer, suffix_manager


class SuffixManager:
    """Suffix manager for adversarial suffix generation."""

    valid_templates = (
        "llama-3",
        "llama-2",
        "vicuna_v1.1",
        "mistral",
        "chatgpt",
        "completion",
        "raw",
        "tinyllama",
        "struq",
        "bipia"
    )

    def __init__(self, *, tokenizer, use_system_instructions, conv_template):
        """Initialize suffix manager.

        Args:
            tokenizer: Tokenizer for model.
            use_system_instructions: Whether to use system instructions.
            conv_template: Conversation template.
        """
        self.tokenizer = tokenizer
        self.use_system_instructions = use_system_instructions
        self.conv_template = conv_template
        self.is_tiktoken = not isinstance(tokenizer, AutoTokenizer)
        logger.info(
            "SuffixManager initialized with conv_template=%s, is_tiktoken=%s, "
            "use_system_instructions=%s",
            self.conv_template.name,
            self.is_tiktoken,
            use_system_instructions,
        )

        self.num_tok_sep = len(
            self.tokenizer(self.conv_template.sep, add_special_tokens=False).input_ids
        )
        if self.conv_template.name == "chatgpt":
            # Space is subsumed by following token in GPT tokenizer
            assert self.conv_template.sep == " ", self.conv_template.sep
            self.num_tok_sep = 0
        elif self.conv_template.name == "llama-3":
            # FastChat adds <|eot_id|> after each message, but it's not sep.
            # Not exactly sure why, but not we need to manually set
            # self.num_tok_sep because sep is just "".
            # https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L167
            self.num_tok_sep = 1
        elif self.conv_template.name == "struq":
            # Somehow "\n\n" sep in Alpaca is tokenized to 3 tokens instead of 2.
            # This requires a manual fix here.
            # \n\n sep is tokenized to 2 in llama3
            self.num_tok_sep = 2 # change to 1 for llama3
        elif self.conv_template.name == "bipia":
            # Somehow "\n\n" sep in Alpaca is tokenized to 3 tokens instead of 2.
            # This requires a manual fix here.
            # \n\n sep is tokenized to 2 in llama3
            self.num_tok_sep = 2 # change to 1 for llama3
        self.num_tok_sep2 = 0
        if self.conv_template.sep2 not in (None, ""):
            self.num_tok_sep2 = len(
                self.tokenizer(self.conv_template.sep2, add_special_tokens=False).input_ids
            )
        if self.conv_template.stop_str not in (None, ""):
            self.num_tok_sep2 += 1

    @torch.no_grad()
    def get_input_ids(
        self,
        messages: list[Message],
        adv_suffix: str | None = None,
        target: str | None = None,
        static_only: bool = False,
    ) -> tuple[torch.Tensor, slice, slice, slice]:
        """Turn messages into token ids.

        Compute token ids for given messages and target, along with slices
        tracking positions of important tokens.

        Args:
            messages: Messages in the conversation.
            adv_suffix: Current adversarial suffix.
            target: Current target output for model.
            static_only: If True, only return token ids for static tokens.

        Returns:
            input_ids: Token ids for messages and target.
            optim_slice: Slice of input_ids corresponding to tokens to optimize.
            target_slice: Slice of input_ids corresponding to target.
            loss_slice: Slice of input_ids corresponding to loss.
        """
        # This code was tested with llama-2 and vicuna_v1.1 templates but remove
        # this check to experiment with others.
        if self.conv_template.name not in self.valid_templates:
            raise NotImplementedError(
                f"{self.conv_template.name} is not implemented! Please use one "
                f"of {self.valid_templates}"
            )
        self.conv_template.messages = []

        if messages[0].content:
            self.conv_template.set_system_message(messages[0].content)

        user_msg = messages[1].content
        if len(messages) <= 2:
            # Toxicity scenario
            self.conv_template.append_message(
                self.conv_template.roles[0], messages[1].content
            )  # user rules
        else:
            if not self.use_system_instructions:
                self.conv_template.append_message(
                    self.conv_template.roles[0], messages[1].content
                )  # user rules
                self.conv_template.append_message(
                    self.conv_template.roles[1], messages[2].content
                )  # asst response
                user_msg = messages[3].content
            # user msg
            self.conv_template.append_message(self.conv_template.roles[0], user_msg)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids

        num_static_tokens = len(toks)
        if user_msg:
            num_static_tokens -= self.num_tok_sep
        elif self.conv_template.name == "vicuna_v1.1":
            pass
        else:
            num_static_tokens -= self.num_tok_sep2

        static_input_ids = torch.tensor(toks[:num_static_tokens])
        if static_only:
            return static_input_ids

        # user msg + adv suffix
        if user_msg:
            if adv_suffix.startswith(" ") and self.is_tiktoken:
                # NOTE: space is part of token in tiktoken, i.e., " !" != "!".
                self.conv_template.update_last_message(f"{user_msg}{adv_suffix}")
            else:
                self.conv_template.update_last_message(f"{user_msg} {adv_suffix}")
        else:
            self.conv_template.update_last_message(adv_suffix)

        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        index = len(toks) - 1 - toks[::-1].index(25782) if (25782 in toks and self.tokenizer.decode([758, 198, 198]) == self.tokenizer.decode([25782])) else -1
        if index != -1: toks.remove(25782); toks.insert(index, 758); toks.insert(index+1, 271) # for llama3
        optim_slice = slice(num_static_tokens, len(toks) - self.num_tok_sep)
        # optim_slice = slice(num_static_tokens, len(toks) - self.num_tok_sep + 1) # llama3, unaltered \n, wrong self.num_tok_sep
        # optim_slice = slice(num_static_tokens + 1, len(toks) - self.num_tok_sep + 2) # llama3, wrong self.num_tok_sep

        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        index = len(toks) - 1 - toks[::-1].index(25782) if (25782 in toks and self.tokenizer.decode([758, 198, 198]) == self.tokenizer.decode([25782])) else -1
        if index != -1: toks.remove(25782); toks.insert(index, 758); toks.insert(index+1, 271) # for llama3
        assistant_role_slice = slice(optim_slice.stop, len(toks))

        self.conv_template.update_last_message(target)  # asst target
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        index = len(toks) - 1 - toks[::-1].index(25782) if (25782 in toks and self.tokenizer.decode([758, 198, 198]) == self.tokenizer.decode([25782])) else -1
        if index != -1: toks.remove(25782); toks.insert(index, 758); toks.insert(index+1, 271) # for llama3
        target_slice = slice(assistant_role_slice.stop, len(toks) - self.num_tok_sep2)
        loss_slice = slice(assistant_role_slice.stop - 1, len(toks) - self.num_tok_sep2 - 1)

        # DEBUG
        #print(self.tokenizer.decode(toks[optim_slice]))
        #print(self.tokenizer.decode(toks[assistant_role_slice]))
        #print(self.tokenizer.decode(toks[target_slice]))
        #print(self.tokenizer.decode(toks[loss_slice]))
        # import pdb; pdb.set_trace()

        # Don't need final sep tokens
        input_ids = torch.tensor(toks[: target_slice.stop])

        return input_ids, optim_slice, target_slice, loss_slice

    @torch.no_grad()
    def gen_eval_inputs(
        self,
        messages: list[Message],
        suffix: str,
        target: str,
        num_fixed_tokens: int = 0,
        max_target_len: int | None = None,
    ) -> EvalInput:
        """Generate inputs for evaluation.

        Returns:
            eval_inputs: Inputs for evaluation.
        """
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids
        suffix_ids.requires_grad_(False)
        suffix_ids.squeeze_(0)

        out = self.get_input_ids(messages, suffix, target)
        orig_input_ids, optim_slice, target_slice, loss_slice = out

        if max_target_len is not None:
            # Adjust target slice to be at most max_target_len
            end = min(target_slice.stop, target_slice.start + max_target_len)
            target_slice = slice(target_slice.start, end)
            loss_slice = slice(loss_slice.start, end - 1)

        # Offset everything to ignore static tokens which are processed
        # separately
        orig_input_ids = orig_input_ids[num_fixed_tokens:]
        optim_slice = slice(
            optim_slice.start - num_fixed_tokens,
            optim_slice.stop - num_fixed_tokens,
        )
        target_slice = slice(
            target_slice.start - num_fixed_tokens,
            target_slice.stop - num_fixed_tokens,
        )
        loss_slice = slice(
            loss_slice.start - num_fixed_tokens,
            loss_slice.stop - num_fixed_tokens,
        )
        target_ids = orig_input_ids[target_slice]
        assert target_ids.ndim == 1
        target_ids.requires_grad_(False)

        eval_input = EvalInput(
            suffix_ids=suffix_ids,
            dynamic_input_ids=orig_input_ids,
            target_ids=target_ids,
            optim_slice=optim_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
        )
        return eval_input


def _simple_template(messages: list[Message]):
    texts = [
        "The following is a conversation between a user and an AI assistant. Please respond to the user as the assistant."
    ]
    for m in messages:
        texts.append(f"{m.role.name.title()}>{m.content}")
    texts.append(f"{Role.ASSISTANT.name.title()}>")
    return "\n".join(texts)


def build_prompt(
    messages: list[Message],
    template_name: str | None = None,
    return_openai_chat_format: bool = False,
):
    if template_name is None:
        return _simple_template(messages)

    conv = get_conv_template(template_name)
    for m in messages:
        if m.role == Role.SYSTEM and m.content:
            conv.set_system_message(m.content)
        elif m.role == Role.USER:
            conv.append_message(conv.roles[0], m.content)
        elif m.role == Role.ASSISTANT:
            conv.append_message(conv.roles[1], m.content)

    # Append assistant response if user message is the last message
    if messages[-1].role == Role.USER:
        conv.append_message(conv.roles[1], None)

    if return_openai_chat_format:
        return conv.to_openai_api_messages()
    return conv.get_prompt()


def batchify_kv_cache(prefix_cache, batch_size):
    batch_prefix_cache = []
    for k, v in prefix_cache:
        batch_prefix_cache.append((k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1, 1)))
    return batch_prefix_cache


def get_nonascii_toks(tokenizer, device="cpu") -> torch.Tensor:
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    non_ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        try:
            tok = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        except:  # noqa: E722, pylint: disable=bare-except
            # GPT tokenizer throws an error for some tokens
            # pyo3_runtime.PanicException: no entry found for key
            continue
        if not is_ascii(tok):
            non_ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        non_ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        non_ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        non_ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        non_ascii_toks.append(tokenizer.unk_token_id)
    non_ascii_toks = list(set(non_ascii_toks))

    return torch.tensor(non_ascii_toks, device=device)


def get_prefix_cache(
    suffix_manager: SuffixManager,
    model,
    tokenizer,
    messages: list[Message],
) -> PrefixCache:
    static_input_ids = suffix_manager.get_input_ids(messages, static_only=True)
    static_input_str = tokenizer.decode(
        static_input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    logger.info("Fixed prefix: %s", static_input_str)
    num_static_tokens = len(static_input_ids)
    logger.info("Fixing the first %d tokens as prefix", num_static_tokens)
    logger.info("Caching prefix...")
    device = model.device if hasattr(model, "device") else model.module.device
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(static_input_ids.to(device)).unsqueeze(0)
        outputs = model(inputs_embeds=input_embeds, use_cache=True)
        prefix_cache = outputs.past_key_values
    return prefix_cache, num_static_tokens
