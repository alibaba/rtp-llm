"""Python wrapper for KVCacheConfig with additional convenience methods."""

import json
from typing import Any, Optional

from rtp_llm.ops import KVCacheConfig as CppKVCacheConfig


class KVCacheConfig(CppKVCacheConfig):
    """Python wrapper for C++ KVCacheConfig with additional convenience methods."""

    def load_prefill_segment_rules(self, input_vocab_size: int) -> None:
        """Resolve deployment rules once, preserving parsed worker snapshots."""
        if self.prefill_segment_before or self.prefill_segment_after:
            return
        path = self.prefill_segment_rules_file
        if not path:
            return
        if input_vocab_size <= 0:
            raise ValueError(f"{path}: input vocabulary size must be positive")
        try:
            with open(path, encoding="utf-8") as reader:
                rules = json.load(reader)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{path}: cannot load prefill segment rules: {error}"
            ) from error
        if not isinstance(rules, dict):
            raise ValueError(f"{path}: expected a JSON object")
        unknown = rules.keys() - {"before", "after"}
        if unknown:
            raise ValueError(f"{path}: unknown rule keys: {sorted(unknown)}")
        parsed: list[list[list[int]]] = []
        for side in ("before", "after"):
            entries = rules.get(side, [])
            if not isinstance(entries, list):
                raise ValueError(f"{path}: {side} must be an array")
            sequences: list[list[int]] = []
            seen: set[tuple[int, ...]] = set()
            for index, entry in enumerate(entries):
                location = f"{path}: {side}[{index}]"
                if not isinstance(entry, list) or not entry:
                    raise ValueError(f"{location} must be a non-empty token ID array")
                for token_index, token in enumerate(entry):
                    if type(token) is not int:
                        raise ValueError(
                            f"{location}[{token_index}] must be an integer token ID"
                        )
                    if not 0 <= token < input_vocab_size or token > 2147483647:
                        raise ValueError(
                            f"{location}[{token_index}] token ID {token} is outside "
                            f"[0, min({input_vocab_size}, 2147483648))"
                        )
                sequence = tuple(entry)
                if sequence not in seen:
                    seen.add(sequence)
                    sequences.append(entry)
            parsed.append(sequences)
        if not any(parsed):
            raise ValueError(f"{path}: at least one before/after rule is required")
        self.prefill_segment_before, self.prefill_segment_after = parsed

    def load_and_update_task_prompt_config(
        self, tokenizer: Optional[Any] = None
    ) -> None:
        """Load task prompt configuration and update token IDs if tokenizer is provided.

        This method combines the functionality of load_task_prompt_config and
        update_task_prompt_tokens_id. It loads the configuration from either
        multi_task_prompt file or multi_task_prompt_str, stores it in
        multi_task_prompt_config, and if a tokenizer is provided, updates the
        token IDs for each task prompt.

        Args:
            tokenizer: Optional tokenizer instance with encode method. If provided,
                     will update task prompt tokens from the loaded configuration.
        """
        # Load task prompt configuration
        if self.multi_task_prompt:
            with open(self.multi_task_prompt, "r") as reader:
                multi_task_prompt_config = json.loads(reader.read(), strict=False)
        elif self.multi_task_prompt_str != "":
            multi_task_prompt_config = json.loads(
                self.multi_task_prompt_str, strict=False
            )
        else:
            return

        # Update task prompt tokens if tokenizer is provided
        if tokenizer and multi_task_prompt_config:
            if isinstance(multi_task_prompt_config, list):
                for info in multi_task_prompt_config:
                    task_id: str = str(info["task_id"])
                    prompt: str = info["prompt"]
                    tokens_id = tokenizer.encode(prompt)
                    self.insertMultiTaskPromptTokens(task_id, tokens_id)

        if self.multi_task_prompt_tokens:
            self.reuse_cache = True
