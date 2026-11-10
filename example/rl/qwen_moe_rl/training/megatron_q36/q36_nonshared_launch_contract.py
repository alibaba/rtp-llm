"""Opt-in gates for the qualified Q36 TP4 copying profile.

Paths are supplied by the operator; contents and live geometry remain checked.
The resulting integration proposal has not itself been run on a GPU.
"""
import hashlib
import importlib.machinery
import json
from pathlib import Path
import sys
from q36_launch_contract import validate as validate_megatron_config

SR_MANIFEST_SHA = 'd824d3ac60b76ac7699dfe36f3935580fdf4033780e3ebb9f8b311e10ac0d4e0'
BRIDGE_SHA = '4fc5b286a9d8569fa6357ce4b69be316b39b6dcb2565d93a2e5d95048935b3f5'
TEXT_FORWARD_SHA = '641852e26d14a60bc237b105b6ee62f9a39470d79aa3cc6f7c1f0ab9d546a6ee'
ADAPTER_SHA = {
    'q36_refit_timing.py': '11f4ea8c99d19d4651e797b9c61a5f59e826897fce2224320d46375a041bf554',
    'rtp_verl_backend.py': '77ba91ba14584b4a996d77d2d6c80a8b7185de413307cdc70532ef469727c599',
    'rtp_native_server.py': '91a9fe875d2f757b9f3c820514311446114bddcaa71d4a83ce9b11c406ec0635',
    'q36_refit_bounded_digest_candidate.py': '118cd27ccb5ab5a6f22e28dba6cc7916584788195f39dd571655ae40844caf43',
}
CUDNN_SHA = {
    'libcudnn.so.9': 'c835893ed01ace9a9cd8089ab981246714c69a5c89af5b97dc8a287d3d0ab03a',
    'libcudnn_adv.so.9': 'a00b98ae905bd4cf67dfade2e8487e3e0edf653c0b6b9c7cff3c3c27d0112164',
    'libcudnn_cnn.so.9': '4a133b1315929dd6b6e14db1ef34e187994ebd7d56884f8795036b135065cf8d',
    'libcudnn_engines_precompiled.so.9': 'f80adfd5e763491d673a7bfdd6775ae211c0f3e9076178e728442eb000e196df',
    'libcudnn_engines_runtime_compiled.so.9': 'aa447551f059224b4f634833d32ff299fb530577f00f41dc25cb93614200071c',
    'libcudnn_engines_tensor_ir.so.9': 'fb2a309a83b55c52e8da9d3a710e8374921e9af407b631ca1506a682455788d7',
    'libcudnn_graph.so.9': '10cc1f9de31263eae09caf0d0f708c888dcfdcc7fbc4994ba2a3f778b0044a80',
    'libcudnn_heuristic.so.9': '0f3b134c38d1050ca6953915e5f75f060b23d8bb3783481f13b78c99e20a46ee',
    'libcudnn_ops.so.9': '0990c59963ae3140e85c517cf31f47614138482559841cc19506e4f54c590401',
    'libcudnn_ext.so.9': '3d525ee3f3063cd337c68f95962009717f8c19294c529ca66f911e13df2b2ba0',
}


def _digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'missing regular file: {path}')
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate(config, env):
    validate_megatron_config(config)
    if (env.get('RTP_RL_SHARED_BACKING', '0') != '0' or
            env.get('RTP_LLM_RL_START_PAUSED', '0') != '0' or
            env.get('RTP_LLM_RL_WORLD72', '0') != '0' or
            env.get('RTP_LLM_RL_MODE') != '1' or env.get('RTP_EP_SIZE') != '1'):
        raise ValueError('this profile requires independent RTP weights and TP4/EP1')
    data = env.get('RLLM_PYTHON_TOOL_TRAIN_JSONL')
    data_sha = env.get('Q36_DATASET_SHA256', '')
    if not data or len(data_sha) != 64 or _digest(data) != data_sha:
        raise ValueError('dataset and independently supplied SHA256 are required')
    with Path(data).open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or len({row['source_row_index'] for row in rows}) != len(rows):
        raise ValueError('dataset must contain distinct, provenance-labelled rows')
    manifest_path = env.get('Q36_SR_MANIFEST')
    if (not manifest_path or env.get('Q36_SR_MANIFEST_SHA256') != SR_MANIFEST_SHA or
            _digest(manifest_path) != SR_MANIFEST_SHA):
        raise ValueError('the qualified live-provider SR manifest is required')
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest.get('binding_scope') != 'full-provider-ddp-live-source-verified' or
            manifest.get('parameter_count') != 20943 or manifest.get('distinct_hf_source_count') != 693):
        raise ValueError('SR manifest coverage mismatch')


def require_real_integrations(env):
    source = Path(env['Q36_CODE_DIR'])
    for name, expected in {
        'q36_source_binding': '486c709867a8039eaefb3b087bfaaa093a606cd012fbf9502a98881ab632808f',
        'q36_training_library_guard': '1fcfb073744e15deb6a363cb5e63ddeaa0ae48d68e1c85975a1a2a6600a4a5a1',
    }.items():
        path = source / (name + '.py')
        spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
        loaded = sys.modules.get(name)
        if (_digest(path) != expected or spec is None or not spec.origin or
                Path(spec.origin).resolve() != path.resolve() or
                loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path.resolve()):
            raise RuntimeError('qualified guard source binding mismatch: ' + name)
    from q36_nonshared_bf16_worker_candidate import verify_candidate_sources
    from q36_training_library_guard import validate_overlay
    validate_overlay(env)
    verify_candidate_sources(env['Q36_OPT_SOURCE_DIR'])
    for name, expected in (('q36_text_bridge_candidate.py', BRIDGE_SHA),
                           ('q36_text_forward_candidate.py', TEXT_FORWARD_SHA)):
        if _digest(source / name) != expected:
            raise RuntimeError('qualified text source mismatch: ' + name)
    adapter = Path(env['Q36_ADAPTER_DIR'])
    if str(adapter) not in env.get('PYTHONPATH', '').split(':'):
        raise RuntimeError('qualified adapter must be present on PYTHONPATH')
    for name, expected in ADAPTER_SHA.items():
        if _digest(adapter / name) != expected:
            raise RuntimeError('qualified adapter mismatch: ' + name)
    from q36_source_binding import verify_source
    for name in ADAPTER_SHA:
        verify_source(name.removesuffix('.py'))
    if env.get('NVTE_FLASH_ATTN') != '0' or env.get('NVTE_FUSED_ATTN') != '1' or env.get('NVTE_UNFUSED_ATTN') != '0':
        raise RuntimeError('qualified fused attention environment required')
    home = Path(env['CUDNN_HOME'])
    if (env.get('LD_PRELOAD') != str(home / 'lib/libcudnn.so.9') or
            env.get('LD_LIBRARY_PATH', '').split(':')[0] != str(home / 'lib')):
        raise RuntimeError('qualified training cuDNN must be preloaded first')
    # Operator-provided pinned library manifest must be authenticated separately.
    manifest = Path(env['Q36_CUDNN_MANIFEST'])
    expected_manifest_sha = env['Q36_CUDNN_MANIFEST_SHA256']
    if len(expected_manifest_sha) != 64 or _digest(manifest) != expected_manifest_sha:
        raise RuntimeError('cuDNN manifest SHA mismatch')
    libraries = json.loads(manifest.read_text())
    if libraries != CUDNN_SHA:
        raise RuntimeError('cuDNN manifest must match the ten qualified libraries')
    for name, expected in libraries.items():
        if Path(name).name != name or len(expected) != 64 or _digest(home / 'lib' / name) != expected:
            raise RuntimeError('cuDNN library mismatch: ' + name)
    launcher = Path(env['RTP_SERVER_LAUNCHER'])
    if _digest(launcher) != env.get('Q36_NATIVE_LAUNCHER_SHA256'):
        raise RuntimeError('native launcher digest mismatch')
