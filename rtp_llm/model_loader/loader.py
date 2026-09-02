import gc
import logging
import math
import os
from collections import OrderedDict
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

import safetensors
import torch
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8PerChannelCompressedQuantConfig
from rtp_llm.lora.lora_weights import LoRAWeights
from rtp_llm.model_loader.ffn_weight import iter_stacked_moe_weights
from rtp_llm.model_loader.load_config import LoadConfig, LoadMethod
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
    ModelWeights,
)
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorCollector
from rtp_llm.model_loader.weight_module import CustomAtomicWeight, WeightModule
from rtp_llm.ops import TaskType, VitSeparation
from rtp_llm.utils.database import (
    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
    BaseDatabase,
    CkptDatabase,
    FastSafeTensorsCompatibilityError,
    _apply_fastsafetensors_env_compat,
    _fastsafetensors_stacked_moe_keyword,
    _normalize_fastsafetensors_stacked_moe_mode,
)
from rtp_llm.utils.model_weight import W, WeightStyle, identity
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.utils.util import check_with_info

# Empirical integration reserve for RTP-owned TensorCollector inputs that
# overlap with final weight materialization. Keep this separate from the
# wrapper-owned estimate and recalibrate with stacked-MoE peak-memory data.
_FASTSAFETENSORS_RTP_COLLECTOR_RESERVE_BYTES = 2 * 1024**3


class ModelLoader:
    WeightInfo = NamedTuple(
        "WeightInfo",
        [
            ("weight", WeightModule),
            ("layer_id", Optional[int]),
            ("collector", TensorCollector),
        ],
    )

    def __init__(
        self,
        model_config: ModelConfig,
        weights_info: ModelDeployWeightInfo,
        misc_weights_info: Optional[CustomAtomicWeight],
        database: BaseDatabase,
        load_method: LoadMethod = LoadMethod.AUTO,
        force_cpu_load_weights: bool = False,
        moe_pure_tp_preshard: bool = False,
    ):
        self.model_config = model_config
        self._task_type = model_config.task_type
        self._load_method = load_method
        self._weights_info = weights_info
        self._misc_weights_info: Optional[CustomAtomicWeight] = misc_weights_info
        if self._misc_weights_info is None:
            self._misc_weights_info = []
        self._model_weights_info: Optional[ModelWeightInfo] = (
            self._weights_info.create_model_weight_info(database)
        )
        # Non-owning global tensors supplied by another live model. Descriptors
        # with these names are excluded before checkpoint iteration and the
        # resulting ModelWeights points directly at the owner's tensors.
        self._global_weight_aliases: Dict[str, torch.Tensor] = {}

        # Get compute_dtype from model_config
        compute_dtype = model_config.compute_dtype
        logging.info(f"load use type {compute_dtype}")

        from rtp_llm.device import get_current_device

        # Get is_attn_model flag from weights_info (calculated in ModelDeployWeightInfo constructor)
        self._is_attn_model = weights_info.is_attn_model
        self._py_eplb, self._phy2log = self.create_eplb()
        self._load_config: LoadConfig = self._weights_info.create_load_config(
            compute_dtype=compute_dtype,
            database=database,
            phy2log=self._phy2log,
            exported_device=get_current_device(),
            force_cpu_load_weights=force_cpu_load_weights,
            moe_pure_tp_preshard=moe_pure_tp_preshard,
        )

    def get_load_config(self) -> LoadConfig:
        return self._load_config

    @property
    def weights_info(self):
        return self._weights_info

    @timer_wrapper(description="load weights")
    @torch.inference_mode()
    def load_weights(
        self,
        device: str,
        global_weight_aliases: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        self._global_weight_aliases = dict(global_weight_aliases or {})
        descriptor_names = {weight.name for weight in self._model_weights_info.weights}
        unknown_aliases = set(self._global_weight_aliases) - descriptor_names
        if unknown_aliases:
            raise KeyError(
                f"global weight aliases are not declared by this model: {sorted(unknown_aliases)}"
            )
        for name, tensor in self._global_weight_aliases.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"global weight alias {name!r} is not a torch.Tensor")
            if tensor.device != torch.device(device):
                raise ValueError(
                    f"global weight alias {name!r} is on {tensor.device}, expected {torch.device(device)}"
                )
        if self._load_config.is_ft_style_weight:
            weights = self._load_from_ft_style(device)
        else:
            weights = self._load_weight(device)
            self.force_clean_cuda_memory()

        # load dynamic weight
        self._load_dynamic_weights(weights, device)
        # load eplb weight
        self._init_eplb_weight(weights, device)
        return weights

    def load_lora_weights(self, adapter_name: str, lora_path: str, device: str = "cpu"):
        lora_weights = LoRAWeights(self._load_config.num_layers)
        # set lora rank
        self._load_config.database.load_lora(adapter_name, lora_path)
        lora_config = self._load_config.database.get_lora_config(adapter_name)
        lora_alpha = lora_config.lora_alpha
        rank = lora_config.rank
        lora_weights.set_lora_rank(rank)
        logging.info(f"load lora weight for adapter {adapter_name}, lora_rank:{rank}")
        if self._weights_info.weight_style == WeightStyle.RTP_LLM_STYLE:
            raise ValueError("load_lora_weights only support non-ft-style weight")

        for id in range(self._load_config.num_layers):
            result = self._load_layer_lora_weights(adapter_name, id, device)
            for name, tensor in result.items():
                lora_weights.set_layer_weight(False, id, name, tensor)

        lora_weights.apply_scale(lora_alpha / rank)  # apply scale
        self._load_config.database.remove_lora(adapter_name)
        return lora_weights

    def dump_weight_as_ft_style(self, device: str, output_dir: str):
        check_with_info(
            not self._load_config.is_ft_style_weight,
            "dump_weight_as_ft_style only support non-ft-style weight",
        )
        tp_rank = self._load_config.tp_rank
        dp_rank = self._load_config.dp_rank
        ep_rank = self._load_config.ep_rank
        weights = self._create_model_weights(device)

        filename_prefix = f"{output_dir}/model-{tp_rank:02d}-{dp_rank:02d}-"
        os.makedirs(output_dir, exist_ok=True)

        max_size = 6 * 1024**3  # 6GB
        part_idx = 0
        current_size = 0
        current_dict = OrderedDict()

        def maybe_save():
            nonlocal current_size, part_idx, current_dict
            if current_size >= max_size:
                filename = f"{filename_prefix}part-{part_idx:05d}.safetensors"
                save_max_retry_times = 2  # maybe fuse is unstable
                for i in range(save_max_retry_times):
                    try:
                        safetensors.torch.save_file(current_dict, filename)
                        logging.info(
                            f"Saved partition {part_idx} ({current_size/1024**3:.2f}GB)"
                        )
                        break
                    except Exception as e:
                        logging.error(f"Failed to save partition {part_idx}: {e}")
                        if i == save_max_retry_times - 1:
                            raise e
                        else:
                            logging.info(
                                f"Failed to save partition {part_idx}: {e}, Retrying..."
                            )
                            continue
                # release gpu memory
                del current_dict
                current_dict = OrderedDict()
                part_idx += 1
                current_size = 0

        for layer_id, name, tensor in self.prepare_weights(device):
            if layer_id is not None:
                tensor_name = f"{weights.layer_weight_prefix(tp_rank, dp_rank, ep_rank)}{layer_id}.{name}"
            else:
                tensor_name = (
                    f"{weights.global_weight_prefix(tp_rank,dp_rank, ep_rank)}{name}"
                )
            tensor_size = tensor.numel() * tensor.element_size()
            current_dict[tensor_name] = tensor.cpu().contiguous()
            current_size += tensor_size
            maybe_save()
            self.force_clean_cuda_memory()

        # save last partition
        if current_dict:
            filename = f"{filename_prefix}part-{part_idx:05d}.safetensors"
            safetensors.torch.save_file(current_dict, filename)
            logging.info(
                f"Saved final partition {part_idx} ({current_size/1024**3:.2f}GB)"
            )
            del current_dict

    @timer_wrapper(description="load_from_ft_style")
    def _load_from_ft_style(self, device: str):
        num_layers = self._load_config.num_layers
        tp_rank = self._load_config.tp_rank
        dp_rank = self._load_config.dp_rank
        ep_rank = self._load_config.ep_rank

        model_weights = ModelWeights(
            num_layers, device, self._load_config.compute_dtype
        )
        layer_weight_prefix = ModelWeights.layer_weight_prefix(
            tp_rank, dp_rank, ep_rank
        )
        global_weight_prefix = ModelWeights.global_weight_prefix(
            tp_rank, dp_rank, ep_rank
        )
        direct_io = self._load_config.exported_device.support_dio_load
        # 清空现有的权重
        weights = [{} for _ in range(num_layers)]
        global_weights = dict(self._global_weight_aliases)
        # 重新构建权重
        all_tensors = self._load_config.database.load_tensors_by_prefix(
            (layer_weight_prefix, global_weight_prefix), device, direct_io=direct_io
        )
        for key, tensor in all_tensors.items():
            if key.startswith(layer_weight_prefix):
                # 解析键名，例如 "layers.0.weight"
                parts = key[len(layer_weight_prefix) :].split(".")
                layer_id = int(parts[0])
                name = ".".join(parts[1:])
                # 将张量移动到设备，并设置到对应的层
                check_with_info(len(tensor) == 1, f"{name} have {len(tensor)} tensor)")
                weights[layer_id][name] = tensor[0].to(device)
            elif key.startswith(global_weight_prefix):
                name = key[len(global_weight_prefix) :]
                if name in self._global_weight_aliases:
                    continue
                check_with_info(len(tensor) == 1, f"{name} have {len(tensor)} tensor)")
                global_weights[name] = tensor[0].to(device)
        model_weights.weights = weights
        model_weights.global_weights = global_weights
        return model_weights

    def _load_weight(self, device: str):
        load_method = self._load_method
        stacked_moe_mode = None
        if load_method == LoadMethod.AUTO:
            is_safetensor = self._load_config.database.is_safetensor
            convert_device = self._choose_weight_convert_device(device)
            tensors_name = self._load_config.database.get_pretrain_tensor_names()
            not_same_name_tensors = len(set(tensors_name)) == len(tensors_name)
            can_try_fastsafetensors = (
                is_safetensor and convert_device != "cpu" and not_same_name_tensors
            )
            if can_try_fastsafetensors:
                requested_mode, stacked_moe_mode, _ = (
                    self._resolve_and_log_fastsafetensors_mode("AUTO")
                )
                if stacked_moe_mode is None:
                    load_method = LoadMethod.SCRATCH
                else:
                    if self._is_memory_enough_for_fastsafetensor(stacked_moe_mode):
                        load_method = LoadMethod.FASTSAFETENSORS
                    else:
                        logging.warning(
                            "AUTO fastsafetensors requested_mode=%s "
                            "effective_mode=scratch degraded_reason="
                            "memory-preflight-failed falls back to scratch",
                            requested_mode,
                        )
                        load_method = LoadMethod.SCRATCH
            else:
                logging.info(
                    "AUTO fastsafetensors requested_mode=auto "
                    "effective_mode=scratch degraded_reason="
                    "prerequisite-failed: is_safetensor=%s convert_device=%s "
                    "unique_tensor_names=%s falls back to scratch",
                    is_safetensor,
                    convert_device,
                    not_same_name_tensors,
                )
                load_method = LoadMethod.SCRATCH

        logging.info(
            f"load method: {load_method}, finally choose load method: {load_method}"
        )

        if load_method.lower() == LoadMethod.FASTSAFETENSORS:
            if stacked_moe_mode is None:
                requested_mode, stacked_moe_mode, _ = (
                    self._resolve_and_log_fastsafetensors_mode("explicit")
                )
                if stacked_moe_mode is None:
                    return self._load_from_scratch(device)
                if (
                    stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
                    and self._has_raw_stacked_moe_weights()
                    and not self._is_memory_enough_for_fastsafetensor(stacked_moe_mode)
                ):
                    logging.warning(
                        "explicit fastsafetensors requested_mode=%s "
                        "effective_mode=scratch degraded_reason="
                        "full-stacked-memory-preflight-failed "
                        "falls back to scratch",
                        requested_mode,
                    )
                    return self._load_from_scratch(device)
            compatibility_error = None
            try:
                return self._load_from_fastsafetensor(device, stacked_moe_mode)
            except FastSafeTensorsCompatibilityError as error:
                # Keep neither the exception nor its traceback alive while
                # scratch loading: the traceback may retain partially loaded
                # model_weights and their GPU tensors.
                compatibility_error = str(error)
            if compatibility_error is not None:
                logging.warning(
                    "fastsafetensors requested_mode=%s effective_mode=scratch "
                    "degraded_reason=runtime-compatibility-failed: %s "
                    "falls back to scratch",
                    stacked_moe_mode,
                    compatibility_error,
                )
                self.force_clean_cuda_memory()
                return self._load_from_scratch(device)
        elif load_method.lower() == LoadMethod.SCRATCH:
            return self._load_from_scratch(device)
        else:
            raise ValueError(f"Unknown load method: {load_method}")

    def _is_memory_enough_for_fastsafetensor(
        self,
        stacked_moe_mode: str = FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
    ):
        model_size = self._weights_info.model_config.eval_model_weight_size()
        device_mem_info = self._load_config.exported_device.get_mem_info()
        max_file_size = self._load_config.database.get_max_file_size()
        if device_mem_info is None:
            return False
        else:
            free_mem = device_mem_info.free / (1024.0**2)
        model_mem = (
            model_size
            / max(self._load_config.ep_size, self._load_config.tp_size)
            / (1024.0**2)
        )
        if self._is_online_ptpc():
            # Online PTPC with inline FP8: MoE expert weights are quantized
            # per-expert during loading (no BF16 peak for MoE), but dense
            # weights (attention, embedding, shared experts) still load as
            # BF16 then quantize to FP8, requiring ~2x peak for that portion.
            moe_params = self._weights_info.model_config.moe_weight_param_count()
            total_layer_params = (
                self._weights_info.model_config.layer_weight_param_count()
            )
            if total_layer_params > 0 and moe_params > 0:
                # dense_ratio: fraction of layer weights that are NOT inline-quantized MoE
                dense_ratio = 1.0 - (moe_params / total_layer_params)
                # Dense weights need 2x (BF16 loaded + FP8 quantized simultaneously),
                # MoE weights only need 1x (quantized inline per-expert).
                # Overall multiplier: dense_ratio * 2 + moe_ratio * 1
                mem_multiplier = dense_ratio * 2.0 + (1.0 - dense_ratio) * 1.0
                model_mem *= mem_multiplier
                logging.info(
                    f"online PTPC with inline FP8: MoE ratio={1.0 - dense_ratio:.2%}, "
                    f"dense ratio={dense_ratio:.2%}, "
                    f"memory multiplier={mem_multiplier:.2f}x (dense needs BF16 peak)"
                )
            else:
                # No MoE weights detected, treat as full online quant
                model_mem *= 2
                logging.info(
                    f"online PTPC but no MoE weights detected, "
                    f"doubling model_mem estimate conservatively"
                )
        elif self._is_online_quant_without_inline():
            # Non-inline online quantization: BF16 checkpoint loaded then
            # quantized to FP8, so peak memory is roughly 2x FP8 model size.
            model_mem *= 2
            logging.info(
                f"online quantization detected (BF16 checkpoint -> FP8), "
                f"doubling model_mem estimate for fastsafetensor memory check"
            )
        max_file_mem = max_file_size / (1024.0**2)
        transient_mem = self._fastsafetensors_transient_budget_bytes(
            max_file_size,
            stacked_moe_mode,
            has_raw_stacked_moe=(
                stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
                and self._has_raw_stacked_moe_weights()
            ),
        ) / (1024.0**2)
        enough = (free_mem - model_mem) > transient_mem
        logging.info(
            f"fastsafetensor memory check: free_mem={free_mem:.0f}MB, "
            f"model_mem={model_mem:.0f}MB, max_file_mem={max_file_mem:.0f}MB, "
            f"transient_mem={transient_mem:.0f}MB, "
            f"stacked_moe_mode={stacked_moe_mode}, enough={enough}"
        )
        return enough

    @staticmethod
    def _fastsafetensors_transient_budget_bytes(
        max_file_size: int,
        stacked_moe_mode: str = FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
        has_raw_stacked_moe: bool = False,
    ) -> int:
        """Return the configured bounded-loader peak or the legacy estimate.

        New fastsafetensors versions expose queue/producer-aware batch-buffer
        accounting via ``estimated_peak_device_bytes``. Keep the historical
        three-shard estimate when that value is unavailable or invalid.
        """
        stacked_moe_mode = _normalize_fastsafetensors_stacked_moe_mode(stacked_moe_mode)
        legacy_budget = 3 * max_file_size
        try:
            from fastsafetensors import load_config

            config = load_config()
            estimate = getattr(config, "estimated_peak_device_bytes", None)
            if estimate is None:
                logging.info(
                    "fastsafetensors does not report a bounded device peak; "
                    "use the legacy three-shard estimate"
                )
                budget = legacy_budget
            elif (
                isinstance(estimate, bool)
                or not isinstance(estimate, (int, float))
                or not math.isfinite(estimate)
                or estimate <= 0
            ):
                logging.warning(
                    "invalid fastsafetensors estimated_peak_device_bytes=%r; "
                    "use the legacy three-shard estimate",
                    estimate,
                )
                budget = legacy_budget
            else:
                budget = int(estimate)
                logging.info(
                    "fastsafetensors memory budget accepts positive upstream "
                    "estimate without applying the legacy floor: "
                    "upstream_estimate_bytes=%d",
                    budget,
                )
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            logging.warning(
                "failed to read bounded fastsafetensors memory config; "
                f"use legacy estimate: {error}"
            )
            budget = legacy_budget
        # The wrapper estimate only accounts for loader-owned buffers. RTP's
        # TensorCollector retains component tensors while weight.load creates
        # the final output. Reserve an empirical 2 GiB for this overlap until
        # model-level stacked-MoE measurements justify a tighter value.
        budget += _FASTSAFETENSORS_RTP_COLLECTOR_RESERVE_BYTES
        if (
            stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
            and has_raw_stacked_moe
        ):
            # The wrapper estimate covers loader-owned buffers. The temporary
            # compatibility path additionally keeps a complete stacked tensor
            # while RTP owns the current expert clone. One max shard is a
            # conservative bound for that integration-owned materialization.
            budget += max_file_size
        logging.info(
            "fastsafetensors memory budget final_budget_bytes=%d "
            "rtp_collector_reserve_bytes=%d stacked_moe_mode=%s "
            "has_raw_stacked_moe=%s",
            budget,
            _FASTSAFETENSORS_RTP_COLLECTOR_RESERVE_BYTES,
            stacked_moe_mode,
            has_raw_stacked_moe,
        )
        return budget

    def _has_raw_stacked_moe_weights(self) -> bool:
        """Return whether this checkpoint needs RTP's full-stacked add-on."""

        _, weight_info_list = self._generate_weight_info()
        return bool(
            self._build_stacked_key_config(weight_info_list, self._load_config.database)
        )

    @staticmethod
    def _fastsafetensors_capability_error(stacked_moe_mode: str) -> Optional[str]:
        """Return why the installed wrapper cannot serve this RTP path."""

        try:
            from fastsafetensors import AutoLoader, SingleGroup
        except ModuleNotFoundError as error:
            return f"package-not-installed: {error}"
        except (AttributeError, ImportError, OSError, RuntimeError) as error:
            return f"package-import-failed: {type(error).__name__}: {error}"
        del SingleGroup
        if stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED:
            return None
        if _fastsafetensors_stacked_moe_keyword(AutoLoader.__init__) is None:
            return (
                "AutoLoader.__init__ is missing stacked_moe_tensors and "
                "legacy dim0_split_templates"
            )
        return None

    @classmethod
    def _resolve_fastsafetensors_mode(
        cls, requested_mode: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve the best supported mode without making package age fatal.

        The returned tuple is ``(effective_mode, reason)``. ``None`` mode means
        the caller must use scratch. A non-None reason with ``full-stacked``
        means bounded per-expert delivery was unavailable but the compatibility
        path remains usable.
        """

        _apply_fastsafetensors_env_compat()
        requested_mode = _normalize_fastsafetensors_stacked_moe_mode(requested_mode)
        base_error = cls._fastsafetensors_capability_error(
            FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED
        )
        if base_error is not None:
            return None, base_error
        if requested_mode == FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT:
            per_expert_error = cls._fastsafetensors_capability_error(requested_mode)
            if per_expert_error is not None:
                return (
                    FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED,
                    per_expert_error,
                )
        return requested_mode, None

    def _resolve_and_log_fastsafetensors_mode(
        self, source: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Resolve package capabilities and emit one actionable decision log."""

        requested_mode = self._fastsafetensors_stacked_moe_mode()
        effective_mode, reason = self._resolve_fastsafetensors_mode(requested_mode)
        if effective_mode is None:
            message = (
                f"{source} fastsafetensors requested_mode={requested_mode} "
                f"effective_mode=scratch degraded_reason={reason} "
                "falls back to scratch"
            )
            if reason and reason.startswith("package-not-installed:"):
                logging.info(message)
            else:
                logging.warning(message)
        elif reason is not None:
            logging.warning(
                "%s fastsafetensors requested_mode=%s effective_mode=%s "
                "degraded_reason=%s",
                source,
                requested_mode,
                effective_mode,
                reason,
            )
        else:
            logging.info(
                "%s fastsafetensors requested_mode=%s effective_mode=%s",
                source,
                requested_mode,
                effective_mode,
            )
        return requested_mode, effective_mode, reason

    @staticmethod
    def _build_stacked_key_config(weight_info_list, database=None) -> dict:
        """Build mapping: stacked ckpt key -> per-expert name template."""
        stacked_key_config = {}
        for wi in weight_info_list:
            for moe_weight in iter_stacked_moe_weights(wi.weight):
                stacked_keys = moe_weight.raw_stacked_tensor_names(wi.layer_id)
                if database is not None and not moe_weight.uses_stacked_expert_keys(
                    database, wi.layer_id
                ):
                    continue
                for idx, (ckpt_weight, stacked_key) in enumerate(
                    zip(moe_weight.weights, stacked_keys)
                ):
                    if ckpt_weight.merge_fun is not identity:
                        continue
                    template = moe_weight._expert_key_pattern(idx).format(
                        i=str(wi.layer_id),
                        expert_id="{expert_id}",
                    )
                    if stacked_key not in stacked_key_config:
                        stacked_key_config[stacked_key] = template
        return stacked_key_config

    @staticmethod
    def _build_fastsafetensors_local_copyout_keys(
        tensor_to_weight_map: Mapping[str, "ModelLoader.WeightInfo"],
        stacked_key_config: Mapping[str, str],
        stacked_moe_mode: str,
    ) -> frozenset[str]:
        """Return checkpoint keys that the current RTP rank can consume.

        Bounded per-expert delivery filters the expanded logical names. The raw
        stacked checkpoint key is retained only for full-stacked delivery,
        where RTP expands it after AutoLoader yields the complete tensor.
        """

        keys = set(tensor_to_weight_map.keys())
        if stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED:
            keys.update(stacked_key_config.keys())
        return frozenset(keys)

    @staticmethod
    def _fastsafetensors_stacked_moe_mode() -> str:
        """Select the stacked-MoE delivery strategy.

        ``per-expert`` preserves the bounded-memory production behavior: the
        source rank slices first and ranks broadcast one expert at a time.
        ``full-stacked`` is the opt-in performance comparison path that
        broadcasts/copies the complete stacked tensor before RTP splits it.
        """

        return _normalize_fastsafetensors_stacked_moe_mode()

    def _is_online_ptpc(self) -> bool:
        quant_config = getattr(self._weights_info, "_quant_config", None)
        return (
            quant_config is not None
            and isinstance(quant_config, Fp8PerChannelCompressedQuantConfig)
            and not quant_config.is_quanted()
        )

    def _is_online_quant_without_inline(self) -> bool:
        """Check if online quantization is active but NOT the inline PTPC path."""
        quant_algo = self._weights_info.model_config.quant_algo
        quant_config = getattr(self._weights_info, "_quant_config", None)
        return (
            quant_algo is not None
            and quant_algo.getWeightBits() == 8
            and quant_config is not None
            and not quant_config.is_quanted()
            and not self._is_online_ptpc()
        )

    def _should_inline_fp8_quantize(self, weight_info) -> bool:
        from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
        from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
            LoadQuantPerChannelFp8Weight,
        )

        weight = weight_info.weight
        if not isinstance(weight, LoadQuantPerChannelFp8Weight):
            return False
        return isinstance(weight.kernel, MoeAtomicWeight) and weight.scale is not None

    def _load_from_fastsafetensor(
        self,
        device: str,
        stacked_moe_mode: str = FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
    ):
        logging.info(f"load weight by device: {device}")
        model_weights = self._create_model_weights(device)
        tensor_to_weight_map, weight_info_list = self._generate_weight_info()

        stacked_key_config = self._build_stacked_key_config(
            weight_info_list, self._load_config.database
        )
        if stacked_key_config:
            logging.info(
                "fastsafetensors stacked MoE mode=%s keys=%d",
                stacked_moe_mode,
                len(stacked_key_config),
            )
            if stacked_moe_mode == FASTSAFETENSORS_STACKED_MOE_MODE_FULL_STACKED:
                logging.warning(
                    "full-stacked is a temporary compatibility fallback: it "
                    "can materially increase peak GPU memory; prefer per-expert "
                    "or use LOAD_METHOD=scratch for the conservative rollback"
                )

        required_checkpoint_keys = self._build_fastsafetensors_local_copyout_keys(
            tensor_to_weight_map, stacked_key_config, stacked_moe_mode
        )
        logging.info(
            "fastsafetensors rank-local copyout filter: keys=%d stacked_keys=%d",
            len(required_checkpoint_keys),
            len(stacked_key_config),
        )

        inline_fp8 = self._is_online_ptpc()
        if inline_fp8:
            from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
                per_channel_cast_to_fp8,
                per_channel_cast_to_fp8_expert,
            )

            logging.info(
                "online PTPC detected: enabling inline FP8 quantization "
                "during fastsafetensors loading to reduce peak GPU memory"
            )

        all_tensors = self._load_config.database.fastsafetensors_weights_iterator(
            device,
            stacked_key_config=stacked_key_config,
            local_copyout_filter=required_checkpoint_keys.__contains__,
            stacked_moe_mode=stacked_moe_mode,
        )

        _inline_count = 0
        _total_count = 0
        for key, loaded_tensor in all_tensors:
            if key not in tensor_to_weight_map:
                continue
            weight_info = tensor_to_weight_map[key]
            _total_count += 1

            if inline_fp8 and self._should_inline_fp8_quantize(weight_info):
                if (
                    loaded_tensor.dtype != torch.float8_e4m3fn
                    and loaded_tensor.dim() == 2
                ):
                    fp8_tensor, scale = per_channel_cast_to_fp8_expert(loaded_tensor)
                    complete = weight_info.collector.store_fp8_quantized(
                        key, fp8_tensor, scale
                    )
                    del loaded_tensor, fp8_tensor, scale
                    _inline_count += 1
                else:
                    complete = weight_info.collector.store_tensor(key, loaded_tensor)
            else:
                complete = weight_info.collector.store_tensor(key, loaded_tensor)

            if inline_fp8 and _total_count % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if _total_count % 5000 == 0 and torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / (1024**3)
                reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                logging.info(
                    f"fastsafetensor loading progress: {_total_count} tensors, "
                    f"{_inline_count} inline-fp8, "
                    f"GPU alloc={alloc_gb:.1f}GiB reserved={reserved_gb:.1f}GiB"
                )
            if complete:
                tensors = weight_info.weight.load(
                    tensor_source=weight_info.collector,
                    layer_id=weight_info.layer_id,
                    device=device,
                    load_config=self._load_config,
                )
                for name, tensor in tensors.items():
                    if weight_info.layer_id is not None:
                        model_weights.set_layer_weight(
                            weight_info.layer_id, name, tensor
                        )
                    else:
                        model_weights.set_global_weight(name, tensor)
                weight_info.collector.clear()
                if inline_fp8:
                    torch.cuda.empty_cache()
                    gc.collect()

        _fallback_count = 0
        for weight_info in weight_info_list:
            weight_info.collector.clear()
            if weight_info.collector.is_collection_complete():
                continue
            _fallback_count += 1
            weight_name = getattr(weight_info.weight, "name", "") or str(
                type(weight_info.weight).__name__
            )
            logging.info(
                f"fastsafetensor fallback: loading {weight_name} "
                f"layer={weight_info.layer_id} from database (collector incomplete)"
            )
            tensors = weight_info.weight.load(
                tensor_source=DatabaseTensorSource(self._load_config.database),
                layer_id=weight_info.layer_id,
                device=device,
                load_config=self._load_config,
            )
            for name, tensor in tensors.items():
                if weight_info.layer_id is not None:
                    model_weights.set_layer_weight(weight_info.layer_id, name, tensor)
                else:
                    model_weights.set_global_weight(name, tensor)
        return model_weights

    def prepare_weights(self, device: str):
        if not self._is_attn_model:
            for id in range(self._load_config.num_layers):
                results = self._load_layer_weights(id, device)
                for name, tensor in results.items():
                    yield (id, name, tensor)

        for weight in self._model_weights_info.weights:
            if self._maybe_skip_weight(weight):
                continue
            weights = weight.load(
                DatabaseTensorSource(self._load_config.database),
                None,
                device,
                self._load_config,
            )
            for name, tensor in weights.items():
                yield (None, name, tensor)

        for weight in self._misc_weights_info:
            weights = weight.load(
                DatabaseTensorSource(self._load_config.database),
                None,
                device,
                self._load_config,
            )
            for name, tensor in weights.items():
                yield (None, name, tensor)

    def _generate_weight_info(self) -> Tuple[Dict[str, WeightInfo], List[WeightInfo]]:
        # WeightInfo = namedtuple("WeightInfo", ["weight", "layer_id", "collector"])
        WeightInfo = ModelLoader.WeightInfo
        tensor_to_weight_map: Dict[str, WeightInfo] = {}
        weight_info_list: List[WeightInfo] = []
        if self._model_weights_info.layer_weights != []:
            for layer_id in range(self._load_config.num_layers):
                layer_weights = self._model_weights_info.layer_weights[layer_id]
                if isinstance(layer_weights, WeightModule):
                    # For CompositeWeight (e.g. MoeWithSharedWeight), split into
                    # sub-components so each gets its own collector. This prevents
                    # large stacked MoE tensors from accumulating in a single
                    # collector waiting for all sub-weights to arrive.
                    for component in layer_weights.get_components():
                        names = component.get_tensor_names(layer_id, self._load_config)
                        collector = TensorCollector(names, self._load_config.database)
                        weight_info = WeightInfo(
                            weight=component, layer_id=layer_id, collector=collector
                        )
                        tensor_to_weight_map.update({k: weight_info for k in names})
                        weight_info_list.append(weight_info)
                else:
                    for weight in layer_weights:
                        for component in weight.get_components():
                            names = component.get_tensor_names(
                                layer_id, self._load_config
                            )
                            collector = TensorCollector(
                                names, self._load_config.database
                            )
                            weight_info = WeightInfo(
                                weight=component, layer_id=layer_id, collector=collector
                            )
                            tensor_to_weight_map.update({k: weight_info for k in names})
                            weight_info_list.append(weight_info)
        for weight in self._model_weights_info.weights:
            if self._maybe_skip_weight(weight):
                continue
            names = weight.get_tensor_names(None, self._load_config)
            collector = TensorCollector(names, self._load_config.database)
            weight_info = WeightInfo(weight=weight, layer_id=None, collector=collector)
            tensor_to_weight_map.update({k: weight_info for k in names})
            weight_info_list.append(weight_info)
        for weight in self._misc_weights_info:
            names = weight.get_tensor_names(None, self._load_config)
            collector = TensorCollector(names, self._load_config.database)
            weight_info = WeightInfo(weight=weight, layer_id=None, collector=collector)
            tensor_to_weight_map.update({k: weight_info for k in names})
            weight_info_list.append(weight_info)
        return tensor_to_weight_map, weight_info_list

    def _maybe_skip_weight(self, weight: WeightModule):
        if weight.name in self._global_weight_aliases:
            return True
        if self._task_type == TaskType.LANGUAGE_MODEL:
            return False
        return weight.name in [W.lm_head]

    @staticmethod
    def force_clean_cuda_memory():
        """安全清理显存，避免残留引用"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @staticmethod
    def force_clean_host_memory():
        """清理host内存，包括Python垃圾回收和glibc malloc缓存"""
        gc.collect()
        # 尝试释放glibc的内存缓存（malloc arena）
        # 这对于释放已经被Python释放但仍被glibc缓存的内存很重要
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            # malloc_trim(0) 会释放所有可以释放的内存回操作系统
            libc.malloc_trim(0)
        except Exception:
            pass  # 某些系统（如macOS）可能不支持

    @staticmethod
    def force_clean_all_memory():
        """清理所有内存（GPU显存和Host内存）"""
        ModelLoader.force_clean_cuda_memory()
        ModelLoader.force_clean_host_memory()

    def cleanup_database(self):
        """清理数据库资源，释放checkpoint加载过程中使用的host内存

        在模型权重加载完成后调用此方法，可以释放以下资源：
        1. CkptFileInfo 中的 metadata 字典（可能包含大量元信息）
        2. pretrain_file_list 和 finetune_file_list 列表
        3. LoRA 相关的缓存数据

        注意：调用此方法后，将无法再从checkpoint加载新的权重，
        但不影响已加载权重的使用和动态LoRA加载功能。
        """
        if self._load_config is None:
            return

        database = self._load_config.database
        if database is None:
            return

        # 清理 CkptFileInfo 的元数据和缓存的 safetensors 句柄
        if database.pretrain_file_list is not None:
            for ckpt_file in database.pretrain_file_list:
                ckpt_file.close_safetensor_handle()
                if ckpt_file.metadata is not None:
                    ckpt_file.metadata = None
            database.pretrain_file_list.clear()

        if database.finetune_file_list is not None:
            for ckpt_file in database.finetune_file_list:
                ckpt_file.close_safetensor_handle()
                if ckpt_file.metadata is not None:
                    ckpt_file.metadata = None
            database.finetune_file_list.clear()

        # 清理 tensor 索引
        if hasattr(database, "_tensor_index"):
            database._tensor_index.clear()

        # 清理 LoRA 缓存
        if database.lora_ckpt is not None:
            database.lora_ckpt = None

        logging.info("Cleaned up database resources to release host memory")

    def _create_model_weights(self, device):
        weights = ModelWeights(
            self._load_config.num_layers, device, self._load_config.compute_dtype
        )
        weights.global_weights.update(self._global_weight_aliases)
        return weights

    def _choose_weight_convert_device(self, current_device):
        if self._load_config.force_cpu_load_weights:
            logging.warning("force_cpu_load_weights is enabled, load weights to cpu")
            return "cpu"
        model_size = self._weights_info.model_config.eval_model_weight_size()
        device_mem_info = self._load_config.exported_device.get_mem_info()
        if device_mem_info is None:
            logging.warning("device_mem_info is None, load weights to cpu")
            return "cpu"
        else:
            free_mem = device_mem_info.free / (1024.0**3)
        model_mem = (
            model_size
            / max(self._load_config.ep_size, self._load_config.tp_size)
            / (1024.0**3)
        )
        device = current_device if free_mem * 0.9 > model_mem else "cpu"
        logging.info(
            f"free_mem: {free_mem:.2f}GB, estimated model_mem: {model_mem:.2f}GB, use device: {device}"
        )
        return device

    def _load_from_scratch(self, device: str):
        weights = self._create_model_weights(device)
        convert_device = self._choose_weight_convert_device(
            device
        )  # choose convert device to avoid out of mem
        logging.info(f"load weight by device: {convert_device}")

        for layer_id, name, tensor in self.prepare_weights(convert_device):
            if convert_device != device:
                tensor = tensor.to(device)
            if layer_id is not None:
                weights.set_layer_weight(layer_id, name, tensor)
            else:
                weights.set_global_weight(name, tensor)
        return weights

    def _load_layer_weights(self, layer_id: int, device: str):
        if self._model_weights_info.layer_weights == []:
            return {}
        assert isinstance(self._model_weights_info.layer_weights[0], list)
        layer_weights = self._model_weights_info.layer_weights[layer_id]
        weights = {}
        for weight in layer_weights:
            res = weight.load(
                DatabaseTensorSource(self._load_config.database),
                layer_id,
                device,
                self._load_config,
            )
            weights.update(res)
        return weights

    def _load_layer_lora_weights(self, lora_name: str, layer_id: int, device: str):
        if self._model_weights_info.layer_weights == []:
            return {}
        assert isinstance(self._model_weights_info.layer_weights[0], list)
        layer_weights = self._model_weights_info.layer_weights[layer_id]
        weights = {}
        for weight in layer_weights:
            res = weight.load_lora(
                self._load_config.database,
                layer_id,
                device,
                self._load_config,
                lora_name,
            )
            weights.update(res)
        return weights

    def _load_dynamic_weights(self, weight: ModelWeights, device: str):
        assert weight is not None, "weight is None"

        embedding_weight = weight.global_weights.get(W.embedding, None)
        if embedding_weight != None:
            self._weights_info.model_config.embedding_size = embedding_weight.shape[0]
            logging.info(
                f"embedding_size is {self._weights_info.model_config.embedding_size}, vocab size is {self._weights_info.model_config.vocab_size}"
            )

        if self._task_type == TaskType.LANGUAGE_MODEL:
            lm_head_w = weight.steal_global_weight(W.lm_head)
            if lm_head_w == None:
                lm_head_w = weight.global_weights[W.embedding]
            if self._weights_info.model_config.normalize_lm_head_weight:
                lm_head_w = F.normalize(lm_head_w)
            logit_scale = self._weights_info.model_config.logit_scale
            if logit_scale != 1.0:
                lm_head_w = logit_scale * lm_head_w
            weight.set_global_weight(W.lm_head, lm_head_w)
        else:
            # Some LLM can be used for other tasks, e.g. classification, in which case lm_head is not needed
            weight.steal_global_weight(W.lm_head)

        pos_weight = weight.global_weights.get(W.positional_embedding, None)
        if pos_weight != None:
            max_seq_len = self._weights_info.model_config.max_seq_len
            if pos_weight.shape[0] < max_seq_len:
                raise Exception(
                    f"positon_weight has shape: {pos_weight.shape}, but max_seq_len is: {max_seq_len} > {pos_weight.shape[0]}"
                )
            pos_weight = pos_weight[:max_seq_len].to(device)
            weight.set_global_weight(W.positional_embedding, pos_weight)

        dynamic_weights = self._weights_info.create_dynamic_weights()
        if dynamic_weights:
            for dynamic_weight in dynamic_weights:
                dynamic_w = dynamic_weight.load(
                    DatabaseTensorSource(self._load_config.database),
                    None,
                    device,
                    self._load_config,
                )
                weight.set_global_weight(
                    dynamic_weight.name, dynamic_w.get(dynamic_weight.name)
                )

    def create_eplb(self):
        weights_info = self._weights_info

        logging.info(
            "create eplb: expert_num: %d, phy_exp_num: %d",
            weights_info.expert_num_,
            weights_info.phy_exp_num_,
        )

        # static expert placement info
        phy2log_path = self.model_config.phy2log_path
        phy2log = LoadConfig.create_redundant_expert(
            layer_num=self.model_config.num_layers,
            expert_num=self.model_config.expert_num,
            ep_size=weights_info.ep_size,
            num_nodes=weights_info.num_nodes,
            phy_exp_num=weights_info.phy_exp_num_,
            phy2log_path=phy2log_path,
        )

        # dynamic expert balancer
        from rtp_llm.eplb.ep_balancer import ExpertBalancer

        model_path = self.model_config.ckpt_path
        ep_lb_database = CkptDatabase(model_path)
        compute_dtype = self.model_config.compute_dtype

        py_eplb = None
        if weights_info.enable_eplb_:
            py_eplb = ExpertBalancer(
                weights_info=weights_info,
                compute_dtype=compute_dtype,
                phy2log=phy2log,
                database=ep_lb_database,
                model_config=self.model_config,
            )
        return py_eplb, phy2log

    def _init_eplb_weight(self, weight: ModelWeights, device: str):
        expert_num = self._load_config.expert_num
        redundant_expert = self._load_config.phy_exp_num - expert_num
        layer_num = self._load_config.num_layers
        phy2log = self._load_config.phy2log

        if expert_num == 0 or (
            not self._weights_info.enable_eplb_ and redundant_expert == 0
        ):
            logging.info("don't need to init eplb weight, skip...")
            return

        # init logic_expert_cnt and log2phy
        for layer_id in range(layer_num):
            logic_expert_cnt = torch.zeros((expert_num,), dtype=torch.int32)
            log2phy = torch.empty(
                (expert_num, redundant_expert + 1), dtype=torch.int32
            ).fill_(-1)
            layer_phy2log = phy2log[layer_id]

            for phy_exp_id, expert_id in enumerate(layer_phy2log):
                cnt = logic_expert_cnt[expert_id]
                log2phy[expert_id, cnt] = phy_exp_id
                logic_expert_cnt[expert_id] += 1

            weight.weights[layer_id][
                W.logic_expert_cnt
            ] = logic_expert_cnt.contiguous().to(device)
            weight.weights[layer_id][W.log2phy] = log2phy.contiguous().to(device)


def get_model_loader(
    model_config: ModelConfig,
    weights_info: ModelDeployWeightInfo,
    misc_weights_info: Optional[CustomAtomicWeight],
    database: BaseDatabase,
    load_method: LoadMethod = LoadMethod.AUTO,
    force_cpu_load_weights: bool = False,
    moe_pure_tp_preshard: bool = False,
) -> ModelLoader:
    if weights_info._head_num % weights_info.tp_size != 0:
        raise Exception(
            "invalid tp_size %d for config.head_num %d"
            % (weights_info.tp_size, weights_info._head_num)
        )
    return ModelLoader(
        model_config,
        weights_info,
        misc_weights_info,
        database,
        load_method=load_method,
        force_cpu_load_weights=force_cpu_load_weights,
        moe_pure_tp_preshard=moe_pure_tp_preshard,
    )
