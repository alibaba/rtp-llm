import gc
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

import safetensors
import torch
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.sleep_mode_compatibility import reject_dynamic_lora_mutation
from rtp_llm.lora.lora_weights import LoRAWeights
from rtp_llm.model_loader.ffn_weight import iter_stacked_moe_weights
from rtp_llm.model_loader.load_config import LoadConfig, LoadMethod
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
    ModelWeights,
)
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorCollector
from rtp_llm.model_loader.weight_memory_saver import (
    is_enabled,
    sleep_mode_level,
    weights_region,
)
from rtp_llm.model_loader.weight_module import CustomAtomicWeight, WeightModule
from rtp_llm.ops import TaskType, VitSeparation
from rtp_llm.utils.database import BaseDatabase, CkptDatabase
from rtp_llm.utils.model_weight import W, WeightStyle, identity
from rtp_llm.utils.module_util import has_module
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.utils.util import check_with_info


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
    ):
        self.model_config = model_config
        self._task_type = model_config.task_type
        self._load_method = load_method
        self._weights_info = weights_info
        self._misc_weights_info: Optional[CustomAtomicWeight] = misc_weights_info
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

        # Dynamic lm_head/positional weights and static EPLB buffers may
        # allocate new GPU tensors outside WeightModule.load().
        with weights_region():
            self._load_dynamic_weights(weights, device)
            self._init_eplb_weight(weights, device)

        # weights_region wraps the whole load pipeline, so the transient
        # intermediates (raw read / dequant / TP-split / .to(device)) it
        # produces get freed here but leave their caching-allocator segments
        # cached. empty_cache() returns those freed segments to the driver so
        # only the live resident weights stay resident -- and, under sleep mode,
        # only they keep the "weights" tag (cudaFree untracks the rest in
        # torch_memory_saver). One-time at load; covers scratch, fastsafetensors
        # and the dynamic/eplb region alike.
        self.force_clean_cuda_memory()
        return weights

    def load_lora_weights(self, adapter_name: str, lora_path: str, device: str = "cpu"):
        reject_dynamic_lora_mutation(
            enable_sleep_mode=is_enabled(), sleep_mode_level=sleep_mode_level()
        )
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

        # Cover LoRA tensors as pausable weight memory when loaded directly to GPU.
        with weights_region():
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
        if load_method == LoadMethod.AUTO:
            is_safetensor = self._load_config.database.is_safetensor
            convert_device = self._choose_weight_convert_device(device)
            tensors_name = self._load_config.database.get_pretrain_tensor_names()
            not_same_name_tensors = len(set(tensors_name)) == len(tensors_name)
            if (
                is_safetensor
                and convert_device != "cpu"
                and not_same_name_tensors
                and self._is_memory_enough_for_fastsafetensor()
                and has_module("fastsafetensors")
            ):
                load_method = LoadMethod.FASTSAFETENSORS
            else:
                load_method = LoadMethod.SCRATCH

        logging.info(
            f"load method: {load_method}, finally choose load method: {load_method}"
        )

        if load_method.lower() == LoadMethod.FASTSAFETENSORS:
            return self._load_from_fastsafetensor(device)
        elif load_method.lower() == LoadMethod.SCRATCH:
            return self._load_from_scratch(device)
        else:
            raise ValueError(f"Unknown load method: {load_method}")

    def _is_memory_enough_for_fastsafetensor(self):
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
        max_file_mem = max_file_size / (1024.0**2)
        logging.debug(
            f"free mem: {free_mem}, model mem: {model_mem}, max file mem: {max_file_mem}"
        )
        return (free_mem - model_mem) > (3 * max_file_mem)

    @staticmethod
    def _build_stacked_key_config(weight_info_list) -> dict:
        """Build mapping: stacked ckpt key -> per-expert name template."""
        stacked_key_config = {}
        for wi in weight_info_list:
            for moe_weight in iter_stacked_moe_weights(wi.weight):
                for idx, ckpt_weight in enumerate(moe_weight.weights):
                    if ckpt_weight.merge_fun is not identity:
                        continue
                    stacked_key = ckpt_weight.tensor_name(wi.layer_id)
                    template = moe_weight._expert_key_pattern(idx).format(
                        i=str(wi.layer_id),
                        expert_id="{expert_id}",
                    )
                    stacked_key_config[stacked_key] = template
        return stacked_key_config

    def _load_from_fastsafetensor(self, device: str):
        model_weights = self._create_model_weights(device)
        # Sleep-mode residual fix: keep the raw fastsafetensors shard reads OUT
        # of the torch_memory_saver "weights" region when the loader always
        # re-materializes each resident weight via a TP/EP/DP split-clone
        # (WeightModule._split -> __split_tensor().contiguous().clone()). The
        # raw shards are allocated by the *iterator* (database.py) under
        # ``allocation_context``, separately from WeightModule.load()'s region;
        # they are pure transients consumed by _split and freed. Routing them
        # through the region's private MemPool strands their freed blocks there
        # forever (empty_cache cannot drain a live private pool, pause skips
        # non-live blocks) -- this is the bulk of the sleep residual. The
        # resident split-clones are still allocated INSIDE weight.load()'s
        # weights_region, so they stay tagged/pausable. Only when every
        # parallelism degree is 1 does _split pass the raw tensor through
        # unchanged (it would then BE the resident weight and must keep the
        # region tag), so fall back to region-scoped raw reads in that case.
        lc = self._load_config
        raw_in_region = lc.tp_size <= 1 and lc.dp_size <= 1 and lc.ep_size <= 1
        for layer_id, name, tensor in self.prepare_weights_fastsafetensor(
            device, in_weights_region=raw_in_region
        ):
            if layer_id is not None:
                model_weights.set_layer_weight(layer_id, name, tensor)
            else:
                model_weights.set_global_weight(name, tensor)
        return model_weights

    def prepare_weights_fastsafetensor(
        self, device: str, in_weights_region: bool = True, force_nogds: bool = False
    ):
        """Bulk fastsafetensors weight generator: yields ``(layer_id, name, tensor)``.

        Streams checkpoint shards through the fastsafetensors iterator and emits
        already-processed tensors (post dequant / MoE per-expert split / TP
        split) — the same layout ``prepare_weights`` produces per-tensor, but via
        the fast bulk shard path instead of per-tensor database reads. Shared by
        the cold load (:meth:`_load_from_fastsafetensor`) and the level-2 wake
        reload (:meth:`WeightManager.reload_weights_from_loader`).

        ``in_weights_region`` controls whether the fastsafetensors allocations are
        scoped into the torch_memory_saver "weights" region:

        * Cold load (``True``): the emitted tensors *become* the resident weights,
          so they must live in the weights region to be paused/resumed by sleep.
        * Wake reload (``False``): the emitted tensors are only transient ``copy_``
          sources into weights that are already resident at a fixed VA. Scoping
          them into the region would commit them (and every dequant/split
          intermediate) as region-backed physical pages that ``cudaFree`` /
          ``empty_cache`` cannot return to the driver — leaving several GB stuck
          (and *growing with weight count*) and starving the subsequent KV-cache
          ``resume`` (observed OOM in ``cu_mem_create``). With ``nullcontext``
          they are plain torch allocations freed per-tensor in the reload loop,
          so the stuck footprint collapses from "scales with the model" to a
          bounded, model-size-independent residual (~1GB order): the freed blocks
          land in torch segments co-tenanted with resident engine allocations, so
          ``empty_cache`` cannot return those segments, but peak simultaneous
          transient is tiny (one tensor at a time) so the residual no longer
          scales. Fully draining that last residual would need per-reload segment
          isolation (a private ``MemPool``), which aborts under
          torch_memory_saver — see ``mempool-destroy-crashes-under-tms``.

        ``force_nogds`` selects the fastsafetensors 'nogds' copier (pread into a
        framework host buffer) over the default 'shm' copier. The level-2 wake
        reload sets it: the 'shm' copier's ``LoadWithShm`` C++ ext faults in
        ``cuMemcpyHtoDAsync_v2`` when it runs after a torch_memory_saver
        pause/resume (its /dev/shm bounce buffer's host registration goes stale
        across the VMM remap). Cold load leaves it ``False`` (shm is faster and
        unaffected).
        """
        logging.info(f"load weight by device: {device}")
        tensor_to_weight_map, weight_info_list = self._generate_weight_info()

        stacked_key_config = self._build_stacked_key_config(weight_info_list)
        if stacked_key_config:
            logging.info(
                f"fastsafetensors per-expert split enabled for {len(stacked_key_config)} stacked keys"
            )

        all_tensors = self._load_config.database.fastsafetensors_weights_iterator(
            device,
            True,
            stacked_key_config=stacked_key_config,
            allocation_context=weights_region if in_weights_region else None,
            force_nogds=force_nogds,
        )

        for key, loaded_tensor in all_tensors:
            if key not in tensor_to_weight_map:
                continue
            weight_info = tensor_to_weight_map[key]

            complete = weight_info.collector.store_tensor(key, loaded_tensor)
            if complete:
                tensors = weight_info.weight.load(
                    tensor_source=weight_info.collector,
                    layer_id=weight_info.layer_id,
                    device=device,
                    load_config=self._load_config,
                )
                for name, tensor in tensors.items():
                    yield (weight_info.layer_id, name, tensor)
                weight_info.collector.clear()

        for weight_info in weight_info_list:
            weight_info.collector.clear()
            if weight_info.collector.is_collection_complete():
                continue
            tensors = weight_info.weight.load(
                tensor_source=DatabaseTensorSource(self._load_config.database),
                layer_id=weight_info.layer_id,
                device=device,
                load_config=self._load_config,
            )
            for name, tensor in tensors.items():
                yield (weight_info.layer_id, name, tensor)

    def can_reload_from_fastsafetensor(self) -> bool:
        """Whether the level-2 wake reload can use the fast bulk fastsafetensors path.

        Distinct from the cold-start check (:meth:`_is_memory_enough_for_fastsafetensor`),
        which sizes headroom for allocating a *second* full copy of the model.
        Wake reload copies into weights that are ALREADY resident (blank pages
        remapped by ``resume``), so only the transient shard buffers need
        headroom — checked against a few max-size files, not the whole model.
        Returns False (caller falls back to the load-from-scratch per-tensor
        reload) when fastsafetensors is unavailable or the checkpoint is not
        fast-loadable (non-safetensors / duplicate tensor names).
        """
        if not has_module("fastsafetensors"):
            logging.info("reload: fastsafetensors module unavailable, use scratch path")
            return False
        if not self._load_config.database.is_safetensor:
            logging.info("reload: checkpoint is not safetensors, use scratch path")
            return False
        tensors_name = self._load_config.database.get_pretrain_tensor_names()
        if len(set(tensors_name)) != len(tensors_name):
            logging.info("reload: duplicate tensor names, use scratch path")
            return False
        device_mem_info = self._load_config.exported_device.get_mem_info()
        if device_mem_info is not None:
            free_mb = device_mem_info.free / (1024.0**2)
            max_file_mb = self._load_config.database.get_max_file_size() / (1024.0**2)
            if free_mb <= 3 * max_file_mb:
                logging.warning(
                    "reload: insufficient transient headroom for fastsafetensors "
                    f"(free={free_mb:.0f}MB <= 3x max_file={3 * max_file_mb:.0f}MB), "
                    "use scratch path"
                )
                return False
        return True

    def prepare_weights(self, device: str):
        if (
            self._load_config.vit_separation != VitSeparation.VIT_SEPARATION_ROLE
            and not self._is_attn_model
        ):
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
        if self._load_config.vit_separation != VitSeparation.VIT_SEPARATION_ROLE:
            for layer_id in range(self._load_config.num_layers):
                layer_weights = self._model_weights_info.layer_weights[layer_id]
                if isinstance(layer_weights, WeightModule):
                    names = layer_weights.get_tensor_names(layer_id, self._load_config)
                    collector = TensorCollector(names, self._load_config.database)
                    weight_info = WeightInfo(
                        weight=layer_weights, layer_id=layer_id, collector=collector
                    )
                    tensor_to_weight_map.update({k: weight_info for k in names})
                    weight_info_list.append(weight_info)
                else:
                    for weight in layer_weights:
                        names = weight.get_tensor_names(layer_id, self._load_config)
                        collector = TensorCollector(names, self._load_config.database)
                        weight_info = WeightInfo(
                            weight=weight, layer_id=layer_id, collector=collector
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
            if (
                layer_id is not None
                and self._load_config.vit_separation
                != VitSeparation.VIT_SEPARATION_ROLE
            ):
                weights.set_layer_weight(layer_id, name, tensor)
            else:
                weights.set_global_weight(name, tensor)
        return weights

    def _load_layer_weights(self, layer_id: int, device: str):
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

        if self._load_config.vit_separation != VitSeparation.VIT_SEPARATION_ROLE:
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

    def prepare_dynamic_weights(self, device: str):
        """Regenerate computed dynamic weights as a ``(layer_id, name, tensor)`` stream.

        The checkpoint-replay generators (:meth:`prepare_weights_fastsafetensor`
        / :meth:`prepare_weights`) only emit tensors that physically exist in the
        checkpoint. A few weights are *computed* at cold load by
        :meth:`_load_dynamic_weights` — e.g. ``rotary_embedding.cos_sin_cache`` —
        and are never read from disk. The level-2 wake reload must regenerate
        them too: after ``resume("weights")`` remaps blank pages at the original
        VA, an uncovered computed weight would stay blank and trip the coverage
        assertion in :meth:`WeightManager.reload_weights_from_loader`. This
        mirrors the ``create_dynamic_weights()`` branch of
        :meth:`_load_dynamic_weights` but yields the tensors (as global weights,
        ``layer_id=None``) instead of writing them into a ``ModelWeights``;
        lm_head / positional_embedding come back through the checkpoint stream.
        """
        if self._load_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE:
            return
        if self._task_type != TaskType.LANGUAGE_MODEL:
            return
        dynamic_weights = self._weights_info.create_dynamic_weights()
        if not dynamic_weights:
            return
        for dynamic_weight in dynamic_weights:
            dynamic_w = dynamic_weight.load(
                DatabaseTensorSource(self._load_config.database),
                None,
                device,
                self._load_config,
            )
            yield (None, dynamic_weight.name, dynamic_w.get(dynamic_weight.name))

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
    )
