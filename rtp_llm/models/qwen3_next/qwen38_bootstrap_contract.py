"""Pinned shape metadata is an allocation contract, never loaded weights."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import math
import os

MODEL_ID = 'Qwen/Qwen3.8-2.4T-A95B'
REVISION = '207bd685a7e3696cfaff12ded7c6a7ea0f88c996'
CONFIG_SHA256 = '4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987'
INDEX_SHA256 = 'e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee'
POLICY_MANIFEST_SHA256 = 'c60d0be889e462b7665922001abfc5a57e3846f33330e33ebfdc044e6b7d3075'
MAIN_NUMEL = 2419804697984


@dataclass(frozen=True)
class BootstrapContract:
    source_shapes: object
    refit_dtypes: object
    checkpoint_dtypes: object
    manifest_sha256: str
    derivation: str


def load_bootstrap_contract(model_root, policy_manifest):
    root = Path(model_root)
    config_bytes = (root/'config.json').read_bytes()
    index_bytes = (root/'model.safetensors.index.json').read_bytes()
    manifest_bytes = Path(policy_manifest).read_bytes()
    for label, data, expected in [('config', config_bytes, CONFIG_SHA256),
                                  ('index', index_bytes, INDEX_SHA256),
                                  ('policy manifest', manifest_bytes, POLICY_MANIFEST_SHA256)]:
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Qwen3.8 bootstrap {label} SHA256 mismatch')
    index = json.loads(index_bytes)['weight_map']
    manifest = json.loads(manifest_bytes)
    if (manifest.get('model_id') != MODEL_ID or manifest.get('revision') != REVISION
            or manifest.get('config_sha256') != CONFIG_SHA256 or manifest.get('index_sha256') != INDEX_SHA256
            or manifest.get('skeleton_generation_admission_allowed') is not False
            or manifest.get('actual_refit_per_key_shape_dtype_and_complete_coverage_required') is not True):
        raise ValueError('Bootstrap metadata identity or admission contract is invalid')
    main = {name for name in index if not name.startswith('mtp.')}
    parameters = manifest['parameters']
    if len(index) != 1609 or len(main) != 1590 or set(parameters) != main:
        raise ValueError('Bootstrap metadata must cover the exact 1590 main-policy sources')
    shapes = {}; refit_dtypes = {}; checkpoint_dtypes = {}
    for name, entry in parameters.items():
        shape = tuple(entry['shape'])
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f'Invalid canonical shape in bootstrap metadata: {name}')
        expected_refit_dtype = 'float32' if name.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias')) else 'bfloat16'
        if (entry.get('checkpoint_file') != index[name] or entry.get('expected_checkpoint_dtype') != 'bfloat16'
                or entry.get('rollout_refit_dtype') != expected_refit_dtype):
            raise ValueError(f'Invalid source file or dtype contract for {name}')
        shapes[name] = shape
        refit_dtypes[name] = expected_refit_dtype
        checkpoint_dtypes[name] = 'bfloat16'
    if sum(math.prod(shape) for shape in shapes.values()) != MAIN_NUMEL:
        raise ValueError('Bootstrap metadata policy parameter count mismatch')
    return BootstrapContract(MappingProxyType(shapes), MappingProxyType(refit_dtypes),
                             MappingProxyType(checkpoint_dtypes), POLICY_MANIFEST_SHA256,
                             manifest['derivation'])


def validate_bootstrap_profile(model, parallel, hardware, moe, env, *, native_capability):
    world32 = env.get('RTP_LLM_RL_WORLD32') == '1'
    if world32 and env.get('RTP_LLM_RL_WORLD72') == '1':
        raise ValueError('Qwen3.8 world32 and world72 are mutually exclusive')
    world = 32 if world32 else 72
    flags = {
        'RTP_LLM_RL_MODE':'1', f'RTP_LLM_RL_WORLD{world}':'1',
        'RTP_LLM_RL_START_PAUSED':'1', 'RTP_LLM_RL_METADATA_BOOTSTRAP':'1',
        'RTP_FIXED_PHYSICAL_EP_LAYOUT':'balanced_v1',
        'RTP_RL_BORROW_SINGLETON_HF_SOURCES':'1',
    }
    if any(env.get(name) != expected for name, expected in flags.items()) or native_capability != 1:
        raise ValueError('Metadata allocation requires the matching native initial-pause capability and all explicit bootstrap flags')
    layout = (parallel.world_size, parallel.tp_size, parallel.dp_size, parallel.ep_size,
              parallel.pp_size, parallel.ffn_sp_size, parallel.get_ffn_tp_size())
    if layout != (world, 8, world//8, world, 1, 8, 1):
        raise ValueError(f'Metadata allocation requires the exact world{world} TP8/DP{world//8}/EP{world} layout')
    if (model.model_type != 'qwen38_moe_text' or model.num_layers != 92 or model.hidden_size != 8192
            or model.expert_num != 512 or model.moe_k != 10 or model.quant_algo.isQuant()
            or str(model.compute_dtype) not in ('torch.bfloat16', 'bfloat16')
            or model.enable_fp32_lm_head or model.normalize_lm_head_weight or model.logit_scale != 1.0
            or model.tie_word_embeddings or model.enable_output_vocab_pruning
            or hardware.use_swizzleA or hardware.enable_cuda_graph or hardware.enable_native_cuda_graph
            or moe.moe_strategy != 'nccl_bf16_reference' or moe.fake_balance_expert):
        raise ValueError('Metadata allocation requires the strict unquantized Qwen3.8 BF16 eager profile')


def loaded_native_library_proof(native, expected_sha256, *, world=72):
    """Verify the exact extension mapped in this process, before allocation."""
    if world not in (32, 72):
        raise ValueError('Unknown Qwen3.8 native world')
    version = getattr(native, f'_rl_world{world}_initial_pause_version', 0)
    if version != 1 or len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Bootstrap requires the accepted native library digest and initial-pause marker')
    path = Path(native.__file__).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(8*1024*1024): digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current) or digest.hexdigest() != expected_sha256:
        raise ValueError('Loaded native library does not match the accepted immutable artifact')
    maps = Path('/proc/self/maps')
    if not maps.exists(): raise ValueError('Native bootstrap proof requires the Linux process map')
    matching = []
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and (fields[5] == str(path) or fields[5] == str(path)+' (deleted)'):
            matching.append(fields)
    if not matching or any(row[5].endswith(' (deleted)') or int(row[4]) != current.st_ino for row in matching):
        raise ValueError('Native library file and current-process mapped inode differ')
    return {'schema':'qwen38-actual-native-library-proof-v1', 'pid':os.getpid(),
            'native_initial_pause_version':version, 'native_library_path':str(path),
            'native_library_sha256':digest.hexdigest(), 'native_library_inode':current.st_ino,
            'native_library_device':current.st_dev, 'native_library_bytes':current.st_size}
