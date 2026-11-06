"""Allocate native target storage without checkpoint payload or model forward.

The allocated bytes are explicitly uninitialized. Native initial-pause gates
must remain closed until a complete real actor-weight transaction commits.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace, MappingProxyType
import importlib
import json
import os

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.models.qwen3_next.qwen38_bootstrap_contract import load_bootstrap_contract, validate_bootstrap_profile, loaded_native_library_proof
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.fixed_physical_ep import validate_native_fixed_profile


class MetadataBootstrapDatabase(CkptDatabase):
    """A CkptDatabase subtype with metadata, and deliberately no payload reader."""
    def __init__(self, contract):
        self.bootstrap_contract = contract
        self._shape_inference_active = False
        self._is_ft_style = False
        self._ft_weight_params = None
        self._recycle_handles = False
        self._tensor_index = {}
        self._file_max_layer = {}
        self.finetune_file_list = []
        self.lora_ckpt = None
        # This is an in-memory index view, never an empty .safetensors file.
        metadata = {name: index for index, name in enumerate(contract.source_shapes)}
        proxy = SimpleNamespace(file_name='<pinned Qwen3.8 policy shape metadata only>', metadata=metadata,
                                get_metadata=lambda: metadata, get_tensor_names=lambda: list(metadata),
                                close_safetensor_handle=lambda: None)
        self.pretrain_file_list = [proxy]

    @property
    def is_safetensor(self): return False

    @property
    def is_ft_style(self): return False

    def has_lora(self): return False

    def has_tensor(self, name): return name in self.bootstrap_contract.source_shapes

    def get_pretrain_tensor_names(self): return list(self.bootstrap_contract.source_shapes)

    def get_tensor_order(self, name): return [list(self.bootstrap_contract.source_shapes).index(name)]

    def get_tensor_type(self, name):
        if not self.has_tensor(name): raise KeyError(name)
        return torch.bfloat16

    def get_max_file_size(self):
        raise RuntimeError('Metadata-only bootstrap has no checkpoint payload files')

    def load_tensor(self, name, data_type=torch.bfloat16):
        if not self._shape_inference_active:
            raise RuntimeError('Metadata-only bootstrap cannot read weights; a real actor refit is required')
        return [torch.empty(self.bootstrap_contract.source_shapes[name], dtype=data_type, device='meta')]

    @contextmanager
    def shape_inference(self):
        if self._shape_inference_active: raise RuntimeError('Nested bootstrap shape inference is forbidden')
        self._shape_inference_active = True
        try: yield
        finally: self._shape_inference_active = False


class _ShapeOnlyCudaPostprocessing:
    # Under the strict CUDA BF16/no-swizzle profile these are also identity in
    # the real CudaImpl/GpuImpl. Do not use this for quantized or ROCm layouts.
    def shuffle_moe_weight(self, tensor, *args): return tensor
    def maybe_rewrite_weight_by_key(self, key, tensor): return tensor


class MetadataBootstrapLoader(ModelLoader):
    def __init__(self, *args, bootstrap_contract, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_contract = bootstrap_contract
        self.bootstrap_source_shapes = bootstrap_contract.source_shapes
        self.bootstrap_source_dtypes = MappingProxyType({name: getattr(torch, dtype) for name, dtype in bootstrap_contract.refit_dtypes.items()})
        self.bootstrap_initialized = False

    def load_weights(self, device, global_weight_aliases=None):
        if global_weight_aliases or self._load_config.merge_lora or self._load_config.is_ft_style_weight:
            raise ValueError('Metadata bootstrap forbids aliases, LoRA and pre-sharded checkpoints')
        if set(self.get_rl_hf_source_names()) != set(self.bootstrap_source_shapes):
            raise ValueError('Native loader source names differ from the pinned bootstrap contract')
        if self._weights_info.create_dynamic_weights():
            raise ValueError('Metadata bootstrap does not support additional value-dependent dynamic descriptors')
        self._global_weight_aliases = {}
        database = self._load_config.database
        shape_config = self._load_config.model_copy(update={'exported_device': _ShapeOnlyCudaPostprocessing()})
        _, groups = self._generate_weight_info()
        specs = {}
        try:
            with database.shape_inference():
                source = DatabaseTensorSource(database)
                for group in groups:
                    converted = group.weight.load(source, group.layer_id, 'meta', shape_config)
                    for name, tensor in converted.items():
                        key = (group.layer_id, name)
                        if key in specs or tensor.device.type != 'meta' or torch._debug_has_internal_overlap(tensor) != 0:
                            raise ValueError('Invalid or overlapping metadata-derived target layout')
                        if tensor.dtype not in (torch.bfloat16, torch.float32) or any(dim <= 0 for dim in tensor.shape):
                            raise ValueError('Invalid metadata-derived target dtype or shape')
                        specs[key] = (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
                    del converted
        finally:
            for group in groups: group.collector.clear()
        if not specs:
            raise ValueError('Metadata bootstrap produced no internal targets')
        storage_bytes = sum((1 + sum((dim-1)*stride for dim, stride in zip(shape, strides))) * torch.empty((),dtype=dtype,device='meta').element_size()
                            for shape, strides, dtype in specs.values())
        if storage_bytes > 96_000_000_000:
            raise ValueError(f'Qwen3.8 target allocation exceeds its explicit 96GB parameter bound: {storage_bytes}')
        weights = self._create_model_weights(device)
        for (layer, name), (shape, strides, dtype) in specs.items():
            value = torch.empty_strided(shape, strides, dtype=dtype, device=device)
            if layer is None: weights.set_global_weight(name, value)
            else: weights.set_layer_weight(layer, name, value)
        self._load_dynamic_weights(weights, device)
        self._init_eplb_weight(weights, device)
        self.bootstrap_target_manifest = {'schema':'qwen38-uninitialized-target-allocation-v1',
            'weights_initialized':False, 'generation_admission_allowed':False,
            'policy_manifest_sha256':self.bootstrap_contract.manifest_sha256,
            'shape_derivation':self.bootstrap_contract.derivation,
            'main_source_count':len(self.bootstrap_source_shapes), 'target_count':len(specs),
            'parameter_storage_bytes':storage_bytes,
            'targets':[{'layer':layer,'name':name,'shape':shape,'stride':strides,'dtype':str(dtype)} for (layer,name),(shape,strides,dtype) in specs.items()]}
        return weights


def create_metadata_bootstrap_loader(model):
    # Fail before any target allocation when a v1 or unrelated library is
    # accidentally mixed with this Python path.
    from rtp_llm.ops import ensure_engine_ops_loaded
    ensure_engine_ops_loaded()
    native = importlib.import_module('libth_transformer')
    validate_bootstrap_profile(model.model_config, model.parallelism_config,
        model.hw_kernel_config, model.moe_config, os.environ,
        native_capability=getattr(native, '_rl_world72_initial_pause_version', 0))
    native_proof = loaded_native_library_proof(native, os.environ.get('RTP_RL_NATIVE_LIBRARY_SHA256', ''))
    validate_native_fixed_profile(model.model_config, model.parallelism_config, model.moe_config, quantized=False)
    if model.merge_lora or model.model_config.lora_infos or model.model_config.ptuning_path or model._weight_alias_names:
        raise ValueError('Metadata bootstrap forbids LoRA, prompt tuning and non-owned weights')
    contract = load_bootstrap_contract(model.model_config.ckpt_path, os.environ['RTP_RL_BOOTSTRAP_POLICY_MANIFEST'])
    database = MetadataBootstrapDatabase(contract)
    weights_info = model.get_weight_cls()(model_config=model.model_config,
        parallelism_config=model.parallelism_config, hw_kernel_config=model.hw_kernel_config,
        kv_cache_config=model.kv_cache_config, merge_lora=False, load_method=model.load_method)
    misc = model.custom_module.get_custom_weight_info() if model.custom_module else []
    if misc: raise ValueError('Metadata bootstrap requires the plain text language model')
    loader = MetadataBootstrapLoader(model.model_config, weights_info, misc, database,
        load_method=model.load_method, force_cpu_load_weights=False,
        moe_pure_tp_preshard=False, bootstrap_contract=contract)
    loader.bootstrap_native_library_proof = dict(native_proof, world_rank=model.parallelism_config.world_rank,
        world_size=72, tp_size=8, dp_size=9, ep_size=72)
    directory = os.environ.get('RTP_WEIGHT_MANIFEST_DIR')
    if directory:
        target = Path(directory)/f'native-library-rank-{model.parallelism_config.world_rank:05d}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+f'.writing-{os.getpid()}')
        temporary.write_text(json.dumps(loader.bootstrap_native_library_proof,indent=2)+'\n')
        temporary.replace(target)
    return loader
