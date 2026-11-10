"""Single-node TP4 RTP entrypoint with the qualified ARM BF16 RoPE backend."""
from pathlib import Path
import hashlib, json, os, runpy, sys

expected={'RTP_LLM_RL_MODE':'1','RTP_LLM_RL_Q36_EP4':'1','WORLD_SIZE':'4','LOCAL_WORLD_SIZE':'4',
          'TP_SIZE':'4','DP_SIZE':'1','EP_SIZE':'4','FFN_SP_SIZE':'4',
          'RTP_ARM_BF16_ROPE_BACKEND':'torch_bf16_v1'}
if any(os.environ.get(k)!=v for k,v in expected.items()):
    raise RuntimeError('Q36 native entrypoint requires the single-node TP4/EP4 BF16 profile')
if any(os.environ.get(k)=='1' for k in ('RTP_LLM_RL_WORLD72','RTP_RL_SHARED_BACKING')):
    raise RuntimeError('Q36 native entrypoint requires normal independent RTP weights')
sys.path.insert(0,os.environ['Q36_ARM_ROPE_DIR'])
from q36_native_source_guard import verify_rope_source
verify_rope_source()
from rtp_arm_bf16_import_hook import install,install_canonical_fusion_guard,verify_rtp_decode_dispatch
verify_rope_source()
backend=install()
if not backend or backend.get('backend')!='torch_bf16_v1' or backend.get('external_rtp_kernel_wheel') is not False:
    raise RuntimeError('wrong BF16 RoPE backend')
from rtp_llm.ops import ensure_engine_ops_loaded
ensure_engine_ops_loaded()
import torch
from q36_native_library_guard import verify_native_mappings, verify_native_search_path
verify_native_search_path(os.environ['LD_LIBRARY_PATH'])
verify_native_mappings(torch.backends.cudnn.version())
fusion=install_canonical_fusion_guard();decode=verify_rtp_decode_dispatch()
if not fusion or not decode:raise RuntimeError('missing canonical fusion or decode dispatch')
from qwen36_overlay import install as install_qwen
install_qwen()
if __name__=='__main__':
    print('Q36_NATIVE_TP4_PREFLIGHT '+json.dumps({'backend':backend,'fusion':fusion,'decode':decode,
        'pid':os.getpid(),'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),flush=True)
    runpy.run_module('rtp_llm.start_server',run_name='__main__')
