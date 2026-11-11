"""Qwen3.8 world entrypoint with the explicit project-owned BF16 RoPE backend.

Module-level installation is intentional: multiprocessing spawn imports this
module again before constructing a backend rank. No pure-TP or EP=1 hook runs.
"""
from pathlib import Path
import hashlib,json,os,runpy,sys,time

HERE=Path(__file__).resolve().parent
ARM=HERE.parent/'arm-rope'
MODE='torch_bf16_v1'
world=32 if os.environ.get('RTP_LLM_RL_WORLD32')=='1' else 72
if world==32 and os.environ.get('RTP_LLM_RL_WORLD72')=='1':
 raise RuntimeError('Qwen3.8 world32 and world72 are mutually exclusive')
expected={
 'RTP_ARM_BF16_ROPE_BACKEND':MODE,
 'RTP_LLM_RL_MODE':'1',f'RTP_LLM_RL_WORLD{world}':'1',
 'RTP_LLM_RL_START_PAUSED':'1','RTP_LLM_RL_METADATA_BOOTSTRAP':'1',
 'RTP_FIXED_PHYSICAL_EP_LAYOUT':'balanced_v1',
 'RTP_RL_BORROW_SINGLETON_HF_SOURCES':'1',
 'WORLD_SIZE':str(world),'LOCAL_WORLD_SIZE':'4','TP_SIZE':'8','DP_SIZE':str(world//8),
 'EP_SIZE':str(world),'FFN_SP_SIZE':'8','PP_SIZE':'1',
 'RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS':str(world//8*8192),
 'PYTORCH_ALLOC_CONF':'backend:native,expandable_segments:False',
}
if any(os.environ.get(key)!=value for key,value in expected.items()):
 raise RuntimeError(f'BF16 RoPE entrypoint requires the frozen world{world} paused profile')
if os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
 raise RuntimeError('Use only the frozen native allocator configuration')
if not (ARM/'rtp_arm_bf16_import_hook.py').is_file():
 raise RuntimeError('The explicit project-owned ARM BF16 RoPE source is missing')
sys.path.insert(0,str(ARM))
from rtp_arm_bf16_import_hook import install,install_canonical_fusion_guard,verify_rtp_decode_dispatch
backend=install()
if not backend or backend.get('backend')!=MODE or backend.get('external_rtp_kernel_wheel') is not False:
 raise RuntimeError('Qwen3.8 requires the explicit BF16 source backend identity')
# Match the public loader order before the canonical CUDA wrapper is imported.
from rtp_llm.ops import ensure_engine_ops_loaded
ensure_engine_ops_loaded()
fusion=install_canonical_fusion_guard()
decode=verify_rtp_decode_dispatch()
if not fusion or not decode:
 raise RuntimeError('The BF16 source backend must retain canonical fusion and explicit decode lengths')
proof=dict(backend,fusion=fusion,decode_dispatch=decode,**{f'world{world}_profile':expected},
           entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           pid=os.getpid(),node_world_rank_env=os.environ.get('RTP_WORLD_RANK'),
           installed_unix=time.time(),pure_tp_hook_called=False,
           external_rtp_kernel_native_library_loaded=False)
if os.environ.get('RTP_WEIGHT_MANIFEST_DIR'):
 target=Path(os.environ['RTP_WEIGHT_MANIFEST_DIR']);target.mkdir(parents=True,exist_ok=True)
 path=target/f'arm-bf16-backend-pid-{os.getpid()}.json'
 temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(proof,indent=2)+'\n');temporary.replace(path)
if __name__=='__main__':
 print(f'RTP_WORLD{world}_EXPLICIT_BF16_ROPE_BACKEND '+json.dumps(proof),flush=True)
 runpy.run_module('rtp_llm.start_server',run_name='__main__')
