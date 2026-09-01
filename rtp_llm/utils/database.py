import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch

from rtp_llm.lora.lora_file import LoraCkpt
from rtp_llm.utils import ckpt_file_info
from rtp_llm.utils.ckpt_file_info import CkptFileInfo, FinetuneType

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.")

FASTSAFETENSORS_STACKED_MOE_MODE_ENV = "RTP_FASTSAFETENSORS_STACKED_MOE_MODE"
FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT = "per-expert"
FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED = "full-stacked"
_FASTSAFETENSORS_STACKED_MOE_MODES = frozenset(
    {
        FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
        FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
    }
)
_FASTSAFETENSORS_NOGDS_CONFIG_JSON = '{"loader":"base","base":{"copier_type":"nogds"}}'
_FASTSAFETENSORS_STACKED_MOE_KEYWORDS = (
    "stacked_moe_tensors",
    "dim0_split_templates",
)
_FASTSAFETENSORS_RUNTIME_COMPATIBILITY_MARKERS = (
    "abi",
    "cannot open shared object file",
    "incompatible fast_safetensors",
    "missing fuse-shm apis",
    "symbol not found",
    "undefined symbol",
)


class FastSafeTensorsCompatibilityError(RuntimeError):
    """An installed wrapper/native combination cannot satisfy RTP's loader API."""


def _apply_fastsafetensors_env_compat() -> None:
    """Apply legacy FastSafeTensors environment compatibility process-wide."""

    if os.environ.get("FASTSAFETENSORS_NOGDS", "0") != "1":
        return
    if (
        os.environ.get("FASTSAFETENSORS_CONFIG_JSON")
        == _FASTSAFETENSORS_NOGDS_CONFIG_JSON
    ):
        return
    os.environ["FASTSAFETENSORS_CONFIG_JSON"] = _FASTSAFETENSORS_NOGDS_CONFIG_JSON
    logging.warning(
        "FASTSAFETENSORS_NOGDS=1 overrides FASTSAFETENSORS_CONFIG_JSON "
        "with the process-wide base/nogds config"
    )


def _normalize_fastsafetensors_stacked_moe_mode(
    mode: Optional[str] = None,
) -> str:
    """Return the transitional stacked-MoE delivery mode."""

    raw_mode = (
        os.environ.get(
            FASTSAFETENSORS_STACKED_MOE_MODE_ENV,
            FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
        )
        if mode is None
        else mode
    )
    normalized = (
        raw_mode.strip()
        if raw_mode and raw_mode.strip()
        else FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
    )
    if normalized not in _FASTSAFETENSORS_STACKED_MOE_MODES:
        raise ValueError(
            f"{FASTSAFETENSORS_STACKED_MOE_MODE_ENV} must be "
            f"'{FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT}' or "
            f"'{FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED}', "
            f"got {normalized!r}"
        )
    return normalized


def _callable_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    """Return whether a callable explicitly accepts a named keyword.

    ``**kwargs`` alone is not a capability declaration: a compatibility
    wrapper may silently discard an unknown optimization keyword.
    """

    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(keyword)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _fastsafetensors_stacked_moe_keyword(callable_obj: Any) -> Optional[str]:
    """Return the supported public stacked-MoE keyword, newest name first."""

    for keyword in _FASTSAFETENSORS_STACKED_MOE_KEYWORDS:
        if _callable_accepts_keyword(callable_obj, keyword):
            return keyword
    return None


def _is_fastsafetensors_compatibility_error(error: BaseException) -> bool:
    """Classify package/API/ABI failures without hiding checkpoint errors."""

    if isinstance(error, (AttributeError, ImportError, ModuleNotFoundError)):
        return True
    if isinstance(error, OSError):
        # dlopen/native ABI failures can surface directly as OSError, but
        # checkpoint paths and storage failures use the same exception family.
        # Only the former are safe to degrade to scratch; data I/O must remain
        # fail-fast so that a second loader does not hide or repeat the fault.
        message = str(error).lower()
        return any(
            marker in message
            for marker in _FASTSAFETENSORS_RUNTIME_COMPATIBILITY_MARKERS
        )
    if isinstance(error, TypeError):
        # Constructor signature drift surfaces as an unexpected/missing argument.
        message = str(error).lower()
        return "argument" in message or "keyword" in message
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        return any(
            marker in message
            for marker in _FASTSAFETENSORS_RUNTIME_COMPATIBILITY_MARKERS
        )
    return False


def _raise_fastsafetensors_compatibility_error(
    context: str, error: BaseException
) -> None:
    if _is_fastsafetensors_compatibility_error(error):
        raise FastSafeTensorsCompatibilityError(
            f"{context}: {type(error).__name__}: {error}"
        ) from error
    raise error


def _iter_fastsafetensors_weights(
    loader: Any,
    stacked_key_config: Optional[Dict[str, str]],
    local_copyout_filter: Optional[Callable[[str], bool]],
) -> Generator[Tuple[str, Any], None, None]:
    """Adapt wrapper output to RTP keys while preserving tensor ownership."""

    for key, tensor in loader.iterate_weights():
        template = (stacked_key_config or {}).get(key)
        if template is None:
            yield key, tensor
            continue

        # MoE/Next checkpoints may store all experts in one tensor
        # [num_experts, ...], while the RTP collectors expect one key per
        # expert. Clone each selected slice because the loader can release the
        # current batch buffer after iteration moves on to the next batch.
        for expert_id in range(tensor.shape[0]):
            expert_key = template.format(expert_id=expert_id)
            if local_copyout_filter is not None and not local_copyout_filter(
                expert_key
            ):
                continue
            yield expert_key, tensor[expert_id].clone()


class BaseDatabase:

    def get_pretrain_tensor_names(self) -> List[str]:
        raise NotImplementedError

    def get_lora_tensor_names(self, name: str) -> List[str]:
        raise NotImplementedError

    def load_tensor(
        self, name: str, data_type: Optional[torch.dtype] = torch.float16
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def has_tensor(self, name: str) -> bool:
        raise NotImplementedError

    def get_tensor_order(self, name: str) -> List[int]:
        raise NotImplementedError

    def get_tensor_type(self, name: str) -> torch.dtype:
        raise NotImplementedError

    def get_max_file_size(self) -> int:
        raise NotImplementedError

    @property
    def is_safetensor(self) -> bool:
        return False

    @property
    def is_ft_style(self) -> bool:
        return False

    @property
    def ft_weight_params(self) -> Optional[Dict[str, Any]]:
        return None


class CkptDatabase(BaseDatabase):

    pretrain_file_list: List[CkptFileInfo]
    finetune_file_list: List[CkptFileInfo]
    lora_ckpt: LoraCkpt

    finetune_type: FinetuneType

    def __init__(
        self,
        path: Optional[str],
        ptuning_path: Optional[str] = None,
        recycle_handles: bool = False,
    ) -> None:

        if path is None:
            return

        self.pretrain_file_list = []
        self.finetune_file_list = []
        self.lora_ckpt = LoraCkpt()
        self._tensor_index: Dict[str, CkptFileInfo] = {}
        self._loaded_layer = -1
        # Safe only because ROCm reads copy out of the mmap; handles reopen lazily.
        self._recycle_handles = recycle_handles and ckpt_file_info.ROCM_COPY_OUT
        self._file_max_layer: Dict[CkptFileInfo, int] = {}

        if os.path.isfile(path):
            raise Exception(f"CkptDatabase needs directory contains checkpoint files")

        self.load_hf_meta(path)
        self._recycle_handles = self._recycle_handles and self.is_safetensor

        self.load_ptuning_meta(ptuning_path)

        self._is_ft_style: bool = self._parse_weight_style(path)

        self._ft_weight_params = (
            self._parse_ft_weight_params(path) if self._is_ft_style else None
        )

        logging.debug(
            f"CkptDatabase all tensor names = {self.get_pretrain_tensor_names()}"
        )

        for ckpt_file in self.pretrain_file_list:
            for tname in ckpt_file.metadata.keys():
                self._tensor_index[tname] = ckpt_file
                if self._recycle_handles and (match := _LAYER_RE.search(tname)):
                    self._file_max_layer[ckpt_file] = max(
                        self._file_max_layer.get(ckpt_file, -1), int(match.group(1))
                    )
        for ckpt_file in self.finetune_file_list:
            for tname in ckpt_file.metadata.keys():
                self._tensor_index[tname] = ckpt_file
        logging.info(
            f"CkptDatabase recycle_handles={self._recycle_handles} (asked={recycle_handles},"
            f" copy_out={ckpt_file_info.ROCM_COPY_OUT}, shards={len(self._file_max_layer)})"
        )
        if self._recycle_handles and not self._file_max_layer:
            logging.warning("recycle_handles on but no layer-numbered tensors; no-op")

    def _recycle_consumed_shards(self, name: str) -> None:
        """Close shards fully below the previous layer (one-layer in-flight slack)."""
        match = _LAYER_RE.search(name) if self._recycle_handles else None
        if match is None:
            return
        layer = int(match.group(1))
        if layer <= self._loaded_layer:
            return
        self._loaded_layer = layer
        for ckpt, max_layer in self._file_max_layer.items():
            if max_layer < layer - 1:
                ckpt.close_safetensor_handle()

    @property
    def is_ft_style(self) -> bool:
        return self._is_ft_style

    @property
    def is_safetensor(self) -> bool:
        return all(map(lambda file: file.is_safetensor(), self.pretrain_file_list))

    @property
    def ft_weight_params(self) -> Optional[Dict[str, Any]]:
        return self._ft_weight_params

    def get_max_file_size(self) -> int:
        if not self.pretrain_file_list:
            return 0
        return max([file.file_size for file in self.pretrain_file_list])

    def filter_by_tensor_name_regexes(
        self, required_tensor_patterns: List[re.Pattern[str]]
    ):
        """Keep only pretrain checkpoint files containing tensors required by the model.

        Finetune files, such as ptuning weights, are intentionally left untouched
        because they are small and still need to be applied after pretrain loading.
        """
        if len(self.pretrain_file_list) <= 1 or not required_tensor_patterns:
            return

        def is_required_file(ckpt_file: CkptFileInfo) -> bool:
            return any(
                pattern.fullmatch(tensor_name)
                for tensor_name in ckpt_file.get_tensor_names()
                for pattern in required_tensor_patterns
            )

        original_count = len(self.pretrain_file_list)
        filtered_file_list = [
            ckpt for ckpt in self.pretrain_file_list if is_required_file(ckpt)
        ]
        if not filtered_file_list:
            logging.warning(
                "filter_by_tensor_name_regexes found no matching checkpoint files; "
                "keep original pretrain_file_list"
            )
            return

        self.pretrain_file_list = filtered_file_list
        logging.info(
            f"filter_by_tensor_name_regexes: {original_count} -> {len(self.pretrain_file_list)} files"
        )

    def load_hf_meta(self, path: str):
        # avoid consolidated.safetensors in Mistral-Nemo-Instruct-2407
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            files = set(json.load(open(index))["weight_map"].values())
            for f in files:
                ckpt = CkptFileInfo(file_name=os.path.join(path, f))
                self.pretrain_file_list.append(ckpt)
            return

        # standard HF
        patterns = ["*.safetensors", "*.bin", "*.pth", "*.pt"]
        glob_files = {}

        for pattern in patterns:
            glob_files[pattern] = [file for file in Path(path).glob(pattern)]

        for _, value in glob_files.items():
            if len(value) != 0:
                exclude_pattern: re.Pattern[str] = re.compile(
                    r".*adapter_model\.bin.*|.*training_args\.bin.*"
                )
                for f in value:
                    if not exclude_pattern.match(f.name):
                        ckpt = CkptFileInfo(file_name=str(f))
                        self.pretrain_file_list.append(ckpt)
                break

    def load_ptuning_meta(self, ptuning_path: Optional[str]):
        if ptuning_path is None or not os.path.exists(ptuning_path):
            return
        for f in Path(ptuning_path).glob("pytorch_model.bin"):
            if not self._contains(f):
                ckpt = CkptFileInfo(
                    file_name=str(f), finetune_type=FinetuneType.ptuning
                )
                self.finetune_file_list.append(ckpt)

    def _contains(self, path: Path):
        for info in self.pretrain_file_list + self.finetune_file_list:
            if Path(info.file_name).resolve() == path.resolve():
                return True
        return False

    def get_pretrain_tensor_names(self) -> List[str]:
        tensor_names = []
        for ckptfile in self.pretrain_file_list:
            tensor_names.extend(ckptfile.get_tensor_names())

        for ckptfile in self.finetune_file_list:
            tensor_names.extend(ckptfile.get_tensor_names())

        return tensor_names

    def load_tensor(
        self, name: str, data_type: Optional[torch.dtype] = torch.float16
    ) -> List[torch.Tensor]:
        ckpt_file = self._tensor_index.get(name)
        if ckpt_file is not None:
            self._recycle_consumed_shards(name)
            return [ckpt_file.load_tensor(name, data_type)]
        return []

    def load_tensor_slice(
        self,
        name: str,
        tensor_slice: Tuple[Union[int, slice], ...],
        data_type: torch.dtype,
    ) -> torch.Tensor:
        ckpt_file = self._tensor_index[name]
        self._recycle_consumed_shards(name)
        return ckpt_file.load_tensor_slice(name, tensor_slice, data_type)

    def get_tensor_shape(self, name: str) -> torch.Size:
        return self._tensor_index[name].get_tensor_shape(name)

    def has_tensor(self, name: str) -> bool:
        return name in self._tensor_index

    def get_tensor_type(self, name: str) -> torch.dtype:
        return self.pretrain_file_list[0].get_tensor_type(name)

    def get_tensor_order(self, name: str) -> List[int]:
        orders = []
        for ckpt_file in self.pretrain_file_list:
            if name in ckpt_file.get_tensor_names():
                orders.append(
                    (ckpt_file.file_name, ckpt_file.get_tensor_read_order(name))
                )

        for ckpt_file in self.finetune_file_list:
            if name in ckpt_file.get_tensor_names():
                orders.append(
                    (ckpt_file.file_name, ckpt_file.get_tensor_read_order(name))
                )

        return orders

    def load_tensors_by_prefix(
        self, prefix_list: List[str], device: str, direct_io: bool
    ) -> dict[str, List[torch.Tensor]]:
        try:
            from fast_safetensors import LoadWithShm

            loader = LoadWithShm(2 * 1024 * 1024 * 1024, device, direct_io)
            load_tensors = lambda ckptfile: loader.load_safetensors_to_device(
                ckptfile.file_name
            )
        except (ModuleNotFoundError, ImportError):
            load_tensors = lambda ckptfile: ckptfile.load_tensors(device, direct_io)

        res = {}
        for ckptfile in self.pretrain_file_list:
            if any(
                tensor.startswith(prefix_list) for tensor in ckptfile.get_tensor_names()
            ):
                tensors = load_tensors(ckptfile)
                for k, v in tensors.items():
                    if not k.startswith(prefix_list):
                        continue
                    if k not in res:
                        res[k] = [v]
                    else:
                        res[k].append(v)
        return res

    def fastsafetensors_weights_iterator(
        self,
        device: str,
        stacked_key_config: Optional[Dict[str, str]] = None,
        local_copyout_filter: Optional[Callable[[str], bool]] = None,
        stacked_moe_mode: str = FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
    ):
        stacked_moe_mode = _normalize_fastsafetensors_stacked_moe_mode(stacked_moe_mode)
        _apply_fastsafetensors_env_compat()
        try:
            from fastsafetensors import AutoLoader, SingleGroup
        except Exception as error:
            _raise_fastsafetensors_compatibility_error(
                "failed to import FastSafeTensors AutoLoader contract", error
            )

        stacked_moe_keyword = _fastsafetensors_stacked_moe_keyword(AutoLoader.__init__)
        if (
            stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
            and stacked_moe_keyword is None
        ):
            logging.warning(
                "fastsafetensors stacked MoE requested_mode=per-expert "
                "effective_mode=full-stacked degraded_reason="
                "AutoLoader.__init__ is missing stacked_moe_tensors and "
                "legacy dim0_split_templates"
            )
            stacked_moe_mode = FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED

        def iterator(device: str):
            if torch.distributed.is_initialized():
                pg = torch.distributed.group.WORLD
            else:
                pg = SingleGroup()

            hf_weights_files = sorted(
                [file.file_name for file in self.pretrain_file_list]
            )
            if device == "cuda":
                device = f"cuda:{pg.rank()}"
                logging.debug(f"origin device is cuda, set to {device}")

            # Backend selection, batching, queue depth, producer count, physical
            # read size and tensor ordering all belong to fastsafetensors. RTP
            # only supplies the process group, files and target device. Standard
            # config entry points are FASTSAFETENSORS_CONFIG_JSON (inline JSON)
            # and FASTSAFETENSORS_CONFIG (JSON file path). The legacy NOGDS
            # compatibility mapping was applied before config probing.
            loader_kwargs: Dict[str, Any] = {}
            if _callable_accepts_keyword(AutoLoader.__init__, "local_copyout_filter"):
                loader_kwargs["local_copyout_filter"] = local_copyout_filter
            elif local_copyout_filter is not None:
                logging.warning(
                    "fastsafetensors copyout requested_mode=rank-local "
                    "effective_mode=consumer-filter degraded_reason="
                    "AutoLoader.__init__ is missing local_copyout_filter; "
                    "materialize all tensors and filter at the RTP consumer"
                )
            effective_copyout_filter = local_copyout_filter
            if (
                stacked_key_config
                and stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
                and local_copyout_filter is not None
            ):
                # The per-expert caller filter excludes raw stacked keys. If a
                # legacy wrapper forces full-stacked delivery, admit those raw
                # keys here and apply the original filter after RTP splits.
                raw_stacked_keys = frozenset(stacked_key_config)
                original_copyout_filter = local_copyout_filter

                def effective_copyout_filter(key: str) -> bool:
                    return key in raw_stacked_keys or original_copyout_filter(key)

                if "local_copyout_filter" in loader_kwargs:
                    loader_kwargs["local_copyout_filter"] = effective_copyout_filter
            if (
                stacked_key_config
                and stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT
            ):
                assert stacked_moe_keyword is not None
                loader_kwargs[stacked_moe_keyword] = stacked_key_config
            try:
                loader = AutoLoader(
                    pg,
                    hf_weights_files,
                    device=device,
                    **loader_kwargs,
                )
            except Exception as error:
                _raise_fastsafetensors_compatibility_error(
                    "failed to construct FastSafeTensors AutoLoader", error
                )
            try:
                try:
                    yield from _iter_fastsafetensors_weights(
                        loader, stacked_key_config, effective_copyout_filter
                    )
                except Exception as error:
                    _raise_fastsafetensors_compatibility_error(
                        "FastSafeTensors iteration compatibility failure", error
                    )
            except BaseException:
                # Cleanup must never replace an active checkpoint, iteration or
                # cancellation failure. Log the secondary close failure and
                # preserve the exception that selected the original semantics.
                try:
                    loader.close()
                except BaseException:
                    logging.warning(
                        "FastSafeTensors close failed while preserving the active error",
                        exc_info=True,
                    )
                raise
            else:
                try:
                    loader.close()
                except Exception as error:
                    _raise_fastsafetensors_compatibility_error(
                        "failed to close FastSafeTensors AutoLoader", error
                    )

        return iterator(device)

    def get_lora_tensor_names(self, config_name: str) -> List[str]:
        return self.lora_ckpt.get_lora_tensor_names(config_name)

    def load_lora_tensor(
        self, lora_name: str, tensor_name: str, data_type: torch.dtype
    ) -> List[torch.Tensor]:
        return self.lora_ckpt.load_lora_tensor(lora_name, tensor_name, data_type)

    def load_lora(self, config_name: str, lora_path: str):
        self.lora_ckpt.load_lora(config_name, lora_path)

    def remove_lora(self, name: str):
        return self.lora_ckpt.remove_lora(name)

    def get_lora_config(self, config_name: str):
        return self.lora_ckpt.get_lora_config(config_name)

    def has_lora(self):
        return self.lora_ckpt.has_lora()

    def get_first_lora_name(self):
        return self.lora_ckpt.get_first_lora_name()

    def dump_lora_info(self) -> None:
        self.lora_ckpt.dump_lora_info()

    def _parse_weight_style(self, ckpt_path: str):
        if ckpt_path and os.path.exists(
            os.path.join(ckpt_path, "model.safetensors.index.json")
        ):
            meta_file = os.path.join(ckpt_path, "model.safetensors.index.json")
            logging.info(f"read weight style from: {meta_file}")
            with open(meta_file, "r") as reader:
                meta_json = json.loads(reader.read())
                return meta_json.get("is_ft_style_weight", False)
        else:
            return False

    def _parse_ft_weight_params(self, ckpt_path: str):
        meta_file = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(meta_file, "r") as reader:
            meta_json = json.loads(reader.read())
            return meta_json.get("__env__params__", None)
