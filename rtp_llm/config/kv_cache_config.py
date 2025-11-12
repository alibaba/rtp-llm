"""Python wrapper for KVCacheConfig with additional convenience methods."""
import json
from typing import Any, Optional

from rtp_llm.ops import KVCacheConfig as CppKVCacheConfig


class KVCacheConfig(CppKVCacheConfig):
    """Python wrapper for C++ KVCacheConfig with additional convenience methods."""

    def load_and_update_task_prompt_config(self, tokenizer: Optional[Any] = None) -> None:
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
        prompt_file_path = self.multi_task_prompt
        if prompt_file_path != "":
            with open(prompt_file_path, "r") as reader:
                self.multi_task_prompt_config = json.loads(reader.read(), strict=False)
        elif self.multi_task_prompt_str != "":
            self.multi_task_prompt_config = json.loads(self.multi_task_prompt_str, strict=False)
        else:
            self.multi_task_prompt_config = None
            return
        
        # Update task prompt tokens if tokenizer is provided
        if tokenizer and self.multi_task_prompt_config:
            multi_task_prompt = self.multi_task_prompt_config
            if isinstance(multi_task_prompt, list):
                for info in multi_task_prompt:
                    task_id: str = str(info["task_id"])
                    prompt: str = info["prompt"]
                    tokens_id = tokenizer.encode(prompt)
                    self.insertMultiTaskPromptTokens(task_id, tokens_id)

