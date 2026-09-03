import contextlib
import fcntl
import functools
import hashlib
import importlib
import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Union


def _purge_dynamic_modules() -> None:
    """Drop every cached ``transformers_modules.*`` module from ``sys.modules``.

    HF's ``get_class_in_module`` inserts the module into ``sys.modules`` *before*
    executing it, so a module that was exec'd from a half-written file stays
    cached in its broken state for the lifetime of the process. Removing the
    entries forces the next ``from_pretrained`` to re-exec it from disk.
    """
    for name in [n for n in sys.modules if n.split(".")[0] == "transformers_modules"]:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _remote_code_lock_dir() -> str:
    """Directory to place the cross-process lock in.

    The lock must live beside the resource it protects -- the shared HF modules
    cache -- not in TMPDIR. Bazel gives every test action its own TMPDIR, so two
    concurrent actions on one node would take two different locks while still
    writing the same modules cache, and serialize against nothing.
    """
    try:
        from transformers.utils import HF_MODULES_CACHE

        if HF_MODULES_CACHE:
            os.makedirs(HF_MODULES_CACHE, exist_ok=True)
            return HF_MODULES_CACHE
    except (ImportError, OSError):
        pass
    fallback = os.path.expanduser("~/.cache/huggingface/modules")
    try:
        os.makedirs(fallback, exist_ok=True)
        return fallback
    except OSError:
        return tempfile.gettempdir()


@contextlib.contextmanager
def _hf_dynamic_module_guard(
    tokenizer_path: str, tokenizer_config: Dict[str, Any]
) -> Iterator[bool]:
    """Serialize HF ``transformers_modules`` cache population across processes.

    Loading a ``trust_remote_code`` tokenizer makes transformers copy the model's
    custom ``tokenization_*.py`` into the shared HF modules cache and then import
    it::

        # transformers/dynamic_module_utils.py, get_cached_module_file()
        if not (submodule_path / module_file).exists() or not filecmp.cmp(...):
            shutil.copy(resolved_module_file, submodule_path / module_file)
        # ... later, in get_class_in_module()
        module_spec.loader.exec_module(module)

    ``shutil.copy`` truncates the destination and refills it, and transformers
    only guards this region with a bare ``threading.Lock`` -- which serializes
    threads but NOT processes. An RTP-LLM server starts the backend ranks, the
    frontend workers and the dash_sc server as separate processes, and each one
    calls ``AutoTokenizer.from_pretrained`` on the same path at roughly the same
    time. A process that imports the file while another is mid-copy execs a
    truncated module, and the tokenizer class silently goes missing::

        AttributeError: module 'transformers_modules.<model>.tokenization_kimi'
                        has no attribute 'TikTokenTokenizer'

    Holding an exclusive file lock across the load closes that window. The lock
    is only taken for models that actually use remote code (``auto_map`` present
    in tokenizer_config.json), so ordinary tokenizers are unaffected.

    Yields True when the lock is held, False when guarding was not needed or the
    lock could not be acquired.
    """
    if not tokenizer_config.get("auto_map"):
        yield False
        return

    digest = hashlib.sha256(
        os.path.realpath(tokenizer_path).encode("utf-8")
    ).hexdigest()[:16]
    lock_path = os.path.join(
        _remote_code_lock_dir(), f"rtp_llm_hf_remote_code_{digest}.lock"
    )
    fd = None
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            # Best effort: keep the lock usable by other users on shared runners.
            os.chmod(lock_path, 0o666)
        except OSError:
            pass
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as e:
        # A lock we cannot take must never block model loading.
        logging.warning(
            f"could not acquire HF remote-code lock {lock_path}: {e}; loading "
            "tokenizer without cross-process serialization"
        )
        if fd is not None:
            os.close(fd)
        yield False
        return
    try:
        yield True
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class BaseTokenizer:
    def __init__(
        self, tokenizer_path: str, config_json: Optional[Dict[str, Any]] = None
    ):
        self.path = tokenizer_path
        self.config_json = config_json or {}
        self.init_tokenizer(tokenizer_path, self.config_json)

    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any]):
        from transformers import AutoTokenizer

        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        tokenizer_obj = None
        if os.path.exists(tokenizer_json_path):
            from tokenizers import Tokenizer as TokenizerFast

            tokenizer_obj = TokenizerFast.from_file(tokenizer_json_path)

        tokenizer_config = self._load_tokenizer_config(tokenizer_path)
        extra_kwargs = self._transformers_v5_kwargs(tokenizer_config, tokenizer_obj)
        extra_kwargs.update(self._additional_kwargs(tokenizer_config))

        def _load():
            return AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                verbose=False,
                use_fast=True,
                **extra_kwargs,
            )

        try:
            with _hf_dynamic_module_guard(tokenizer_path, tokenizer_config) as guarded:
                try:
                    self.tokenizer = _load()
                except AttributeError as e:
                    # A previously cached, half-executed remote-code module poisons
                    # sys.modules for the rest of the process. Purge and retry once
                    # -- by now the lock guarantees the file on disk is complete.
                    if "transformers_modules" not in str(e):
                        raise
                    logging.warning(
                        f"remote code module for {tokenizer_path} was incomplete "
                        f"({e}); purging transformers_modules cache and retrying "
                        f"(guarded={guarded})"
                    )
                    _purge_dynamic_modules()
                    self.tokenizer = _load()
        except Exception as e:
            logging.error(
                f"AutoTokenizer.from_pretrained failed for tokenizer_path={tokenizer_path}, "
                f"extra_kwargs={extra_kwargs}: {e}"
            )
            raise
        self._fix_post_processor(tokenizer_obj, extra_kwargs)

    def _additional_kwargs(self, tokenizer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses to inject extra kwargs before from_pretrained."""
        return {}

    @staticmethod
    def _load_tokenizer_config(tokenizer_path: str) -> Dict[str, Any]:
        """Parse tokenizer_config.json once (returns {} if absent)."""
        config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"failed to parse {config_path}: {e}") from e

    @staticmethod
    def _transformers_v5_kwargs(
        tokenizer_config: Dict[str, Any], tokenizer_obj=None
    ) -> Dict[str, Any]:
        """Workaround for transformers==5.2.0 from_pretrained regressions.

        Transformers 5.2.0 rewrote tokenizer loading via TokenizersBackend. Two issues
        require explicit kwargs to preserve correct behavior:

        1. add_eos_token / add_bos_token (found on gte-Qwen2-7B-instruct and
           DeepSeek-V2-Lite-Chat respectively):
           from_pretrained no longer passes these from tokenizer_config.json to
           custom tokenizer __init__. The custom class falls back to its default
           (False), so BOS/EOS is not appended during encode — breaking embedding
           models (last-token pooling) and chat models (missing BOS changes output).
           Fix: explicitly pass add_eos_token/add_bos_token from tokenizer_config.json.
           NOTE: upstream main (fd6bc380c8) intentionally pops these when
           tokenizer.json exists, expecting post_processor to handle it — but models
           like gte-Qwen2 have no EOS in post_processor. This workaround is needed
           long-term unless the model's tokenizer.json is updated.

        2. tokenizer_object (affects models with tokenizer_class: LlamaTokenizerFast,
           e.g. DeepSeek-R1 series):
           Class-specific __init__ (LlamaTokenizer) unconditionally rebuilds the
           internal _tokenizer with Metaspace pre_tokenizer, overriding what
           tokenizer.json defines (e.g. regex Split). This causes whitespace/newlines
           to be silently dropped during encode ("\\n\\n" -> []).
           Fix: pass tokenizer_object loaded directly from tokenizer.json.
           TokenizersBackend.__init__ uses this object to overwrite the class-built
           _tokenizer, preserving the correct pre_tokenizer/decoder from the file.
           NOTE: upstream tracks this as huggingface/transformers#45488.
           Our fix is model-agnostic and stable — keep until upstream is reliable.
        """
        kwargs: Dict[str, Any] = {}
        if "add_eos_token" in tokenizer_config:
            kwargs["add_eos_token"] = tokenizer_config["add_eos_token"]
        if "add_bos_token" in tokenizer_config:
            kwargs["add_bos_token"] = tokenizer_config["add_bos_token"]

        if tokenizer_obj is not None:
            kwargs["tokenizer_object"] = tokenizer_obj
        return kwargs

    def _fix_post_processor(self, tokenizer_obj, extra_kwargs):
        """Workaround for transformers==5.2.0 post_processor override.

        Transformers 5.2.0 tokenizer classes (e.g. XLMRobertaTokenizer) overwrite
        the post_processor in __init__ with a hardcoded template AFTER super().__init__()
        has already restored the correct one from tokenizer_object. This means
        tokenizer.json's post_processor (e.g. double </s></s> for RoBERTa pair inputs)
        is lost and replaced by the class's default (single </s>).

        Fix: two-phase restore:
        1. Unconditionally restore post_processor from tokenizer.json (undoes class
           __init__ corruption).
        2. If add_eos_token/add_bos_token was passed in extra_kwargs, call
           update_post_processor() to re-inject BOS/EOS via transformers' standard
           mechanism. This rebuilds a TemplateProcessing that includes the special tokens.
        """
        import transformers
        from packaging import version

        if version.parse(transformers.__version__).major < 5:
            return
        if tokenizer_obj is None:
            return
        if not hasattr(self.tokenizer, "_tokenizer"):
            # slow tokenizer, nothing to restore.
            return
        if tokenizer_obj.post_processor is not None:
            self.tokenizer._tokenizer.post_processor = tokenizer_obj.post_processor
        # If tokenizer.json's post_processor disagrees with tokenizer_config's add_eos/bos_token,
        # tokenizer.json wins: rebuild below only when add_eos/bos_token is explicitly True.
        if (
            extra_kwargs.get("add_eos_token") or extra_kwargs.get("add_bos_token")
        ) and hasattr(self.tokenizer, "update_post_processor"):
            self.tokenizer.update_post_processor()

    def encode(self, prompt: str, **kwargs):
        return self.tokenizer.encode(prompt, **kwargs)

    def decode(self, token_id: Union[int, List[int]], **kwargs):
        if isinstance(token_id, List) and len(token_id) == 0:
            return ""
        return self.tokenizer.decode(token_id, **kwargs)

    def batch_decode(self, token_ids: Union[List[int], List[List[int]]], **kwargs):
        return [
            self.tokenizer._decode(
                seq,
                **kwargs,
            )
            for seq in token_ids
        ]

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    @property
    def stop_words_id_list(self):
        return []

    @property
    def stop_words_str_list(self):
        return []

    @property
    def chat_template(self):
        return self.tokenizer.chat_template

    @property
    def default_chat_template(self):
        return self.tokenizer.default_chat_template

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        if self.tokenizer.eos_token_id is None:
            return self.config_json.get("eos_token_id") or 0
        else:
            return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token(self):
        return self.tokenizer.bos_token

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    def get_real_tokenizer(self):
        return self.tokenizer

    def tokenize(self, text: str, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)

    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    @property
    def _added_tokens_encoder(self):
        return self.tokenizer._added_tokens_encoder

    @property
    def vocab_size(self):
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        else:
            return self.config_json.get("vocab_size", 0)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: List[str]):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)

    @property
    def is_fast(self):
        return self.tokenizer.is_fast

    def get_added_vocab(self):
        return self.tokenizer.get_added_vocab()

    @property
    def model_max_length(self):
        return self.tokenizer.model_max_length

    @property
    def special_tokens_map(self):
        return self.tokenizer.special_tokens_map

    def save_pretrained(self, save_directory, **kwargs):
        return self.tokenizer.save_pretrained(save_directory, **kwargs)

    @property
    def additional_special_tokens(self):
        try:
            return self.tokenizer.additional_special_tokens
        except AttributeError:
            return getattr(self.tokenizer, "extra_special_tokens", [])

    def add_special_tokens(
        self,
        special_tokens_dict: Dict[str, Any],
        replace_additional_special_tokens: bool = True,
    ):
        return self.tokenizer.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens
        )

    def __str__(self) -> str:
        return self.tokenizer.__str__()

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)

    @functools.cache
    def __len__(self) -> int:
        return self.tokenizer.__len__()
