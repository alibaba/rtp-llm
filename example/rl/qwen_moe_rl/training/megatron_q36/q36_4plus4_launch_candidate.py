"""Frozen 4+4 recipe preflight and explicitly supervised launcher candidate.

Preflight is CPU-only. The `run` mode launches a real training driver and must
only be used after the old task is fully released and a fresh claim exists.
"""
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCES = {
 'Q36_CODE_DIR': {
  'q36_4plus4_recipe.py': '3944fd0bb0c7fc613507f54006cdb1592fe67fbb37e5563825f6d61924b2f4cb',
  'run_standalone_q36_candidate_v3.py': 'c3a9d610f593b1baa84e6e4c7033bbc7bd0e4e97bc920b63d8095c670e764316',
  'onpolicy_standalone_candidate.py': '8c36ff1bb662fa5ba5177c1f4e17646664c81fe93790087f00d3c10178f468a8',
  'q36_4plus4_claim_loader.py': '0075f76555d41822bfd23d3879165aa754b0b0f19484f99ec3f5166dfbe0c5ac',
  'q36_4plus4_claim_guard.py': '8bbfcd82128c1c2ac4018ec1f9fd101547389523077acbbd4c3435c5321f9e0e',
  'q36_standalone_bootstrap_selector.py': 'f921d6b4fc82246fc702d2afcec278ae2d4c763924539200ac3ecf409dd34c6e',
  'standalone_bootstrap_rpc_v6.py': 'e8caff16e23780db8ec592fa050d6ead8fce6de0395dbd4ff2ac3ec2d25ff02c',
  'standalone_bootstrap_v4.py': 'd3c993deb80505b578b97fd5222ff396dec4f0b5362dc6937aa503ff211e66a8'},
 'Q36_ADAPTER_DIR': {
  'standalone_rtp_adapter_v3.py': '746017e37515ecacbbd17e822d8665a619a21fd04b27ca327b246890cbca6c35',
  'standalone_rtp_adapter_v4.py': '2b918782db5107752212184343afd0cf4710f99a6b841147071151163e4cdb71',
  'rtp_cleanup.py': '4506fe79022d0d02f14118ee30d65f1a935a4bb63e51d538831287fb663bdb5b'},
 'Q36_DISAGG_CODE_DIR': {
  'verl_nixl_guarded.py': 'fbd10229677060126e09a9840fa4b34a758ea76170d965daa39e45839a609237',
  'full_hf_relay.py': '0043b7fee1b6e80551174460cb1401aa2eb6c4216f346778c202984ab1fe55af'}}
MODEL_SHA = {'config.json': '93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99',
             'model.safetensors.index.json': '41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83'}
REQUIRED_FLAGS = {'Q36_STRICT_4PLUS4_CLAIM':'1', 'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1',
 'Q36_USE_RPC_OVERLAY':'1',
 'Q36_DISAGG_CLEANUP_SINGLE_NODE_TP4':'1', 'Q36_DISAGG_TRAIN_GPUS':'4',
 'Q36_DISAGG_ROLLOUT_GPUS':'4', 'RTP_EP_SIZE':'1', 'RTP_RL_SHARED_BACKING':'0',
 'RTP_TP_SIZE':'4', 'RLLM_RTP_GPUS_PER_NODE':'4', 'RLLM_RTP_EXPECTED_REPLICAS':'1',
 'RLLM_REQUIRE_REAL_UPDATE':'1', 'NVTE_FLASH_ATTN':'0', 'NVTE_FUSED_ATTN':'1',
 'NVTE_UNFUSED_ATTN':'0', 'UCX_MODULES':'^efa', 'UCX_IB_ROCE_REACHABILITY_MODE':'all'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(env, sources=SOURCES):
    for root_name, entries in sources.items():
        root = Path(env[root_name]).resolve(strict=True)
        for name, expected in entries.items():
            source = root / name
            if source.is_symlink() or not source.is_file() or digest(source) != expected:
                raise RuntimeError('fixed source mismatch: ' + str(source))
            spec = importlib.machinery.PathFinder.find_spec(source.stem, sys.path)
            if spec is None or not spec.origin or Path(spec.origin).resolve() != source:
                raise RuntimeError('Python would import a different source: ' + source.stem)
            loaded = sys.modules.get(source.stem)
            if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source:
                raise RuntimeError('Python already imported a different source: ' + source.stem)


def preflight(env):
    for key, expected in REQUIRED_FLAGS.items():
        if env.get(key) != expected:
            raise RuntimeError('required environment mismatch: ' + key)
    if env.get('UCX_TLS') != 'rc,self,sm,cuda_copy,cuda_ipc':
        raise RuntimeError('qualified RoCE UCX transport list is required')
    run = Path(env['Q36_RUN_ROOT']).resolve(strict=True)
    acceptance = Path(env['RLLM_RTP_ACCEPTANCE_DIR']).resolve(strict=True)
    if not acceptance.is_relative_to(run) or acceptance == run:
        raise RuntimeError('acceptance directory must belong to this run')
    ready_path = run / 'ray-claim-ready.json'
    if ready_path.is_symlink() or not ready_path.is_file():
        raise RuntimeError('both nodes must have a verified claim before launch')
    ready = json.loads(ready_path.read_bytes())
    expected_peer = {'node_ip': '10.44.33.31', 'claim_path': str(run / 'ray-claim.json'),
                     'claim_sha256': env['Q36_RAY_CLAIM_SHA256']}
    if (ready.get('run_id') != run.name or ready.get('claim_path') != str(run / 'ray-claim.json') or
        ready.get('claim_sha256') != env['Q36_RAY_CLAIM_SHA256'] or
        ready.get('training_node_ip') != '10.44.49.51' or ready.get('rollout_readback') != expected_peer):
        raise RuntimeError('claim replication receipt differs from this run')
    verify_sources(env)
    from q36_standalone_bootstrap_selector import select_bootstrap
    rpc_bootstrap = select_bootstrap()
    if rpc_bootstrap.__name__ != 'standalone_bootstrap_rpc_v6':
        raise RuntimeError('4+4 requires the RPC-aware standalone worker hook')
    rpc_bootstrap._install_proto()
    rpc_bootstrap._load_v5()._verify_v5()
    resident = env.get('Q36_DISAGG_RESIDENT_KV', '0')
    if resident not in ('0', '1'):
        raise RuntimeError('Q36_DISAGG_RESIDENT_KV must be 0 or 1')
    if resident == '1':
        if env.get('RLLM_RTP_KEEP_KV_CACHE') != '1' or env.get('RTP_LLM_RL_MODE') != '1':
            raise RuntimeError('resident KV requires explicit native keep and RL mode')
        verify_sources(env, {
          'Q36_CODE_DIR': {'standalone_bootstrap_v5.py':'1e25f1b0f9f546173fd623413dfcf63545af1d53bf155c419e3fbfd69c5d448c'},
          'Q36_ADAPTER_DIR': {'standalone_rtp_adapter_v5.py':'b1adfbd8a300197568af4c3b6706e35bc46f34f07b4e368571b0f697354e10e1'}})
    import rtp_cleanup
    if rtp_cleanup._qualified_owner_count() != 1:
        raise RuntimeError('cleanup must select exactly one qualified native owner')
    model = Path(env['Q36_MODEL_PATH'])
    for name, expected in MODEL_SHA.items():
        if digest(model / name) != expected:
            raise RuntimeError('mounted model identity differs: ' + name)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('preflight must not initialize CUDA')
    import hydra
    from run_standalone_q36_candidate_v3 import validate
    from q36_4plus4_recipe import overrides
    with hydra.initialize_config_module(version_base=None, config_module='rllm.trainer.config'):
        cfg = hydra.compose(config_name='unified', overrides=overrides())
    validate(cfg)
    import verl_nixl_guarded
    if verl_nixl_guarded.RtpGuardedNixl.support_publish_weights is not False:
        raise RuntimeError('checkpoint completion semantics changed')
    if torch.cuda.is_initialized():
        raise RuntimeError('preflight unexpectedly initialized CUDA')
    return {'run_id': run.name, 'claim_sha256': env['Q36_RAY_CLAIM_SHA256'],
            'checkpoint_engine': 'rtp_nixl_guarded', 'support_publish_weights': False,
            'cuda_initialized': False, 'ray_started': False}


def run(env):
    import psutil
    checked = preflight(env)
    from q36_4plus4_recipe import overrides
    run_root = Path(env['Q36_RUN_ROOT']).resolve(strict=True)
    log = run_root / 'training-driver.raw.log'
    result = run_root / 'training-driver.result.json'
    identity_path = run_root / 'training-driver.identity.json'
    if any(path.exists() or path.is_symlink() for path in (log, result, identity_path)):
        raise FileExistsError('existing driver evidence prohibits replay')
    driver = str(Path(env['Q36_CODE_DIR']) / 'run_standalone_q36_candidate_v3.py')
    read_fd, write_fd = os.pipe()
    gate = ('import os,sys; ok=os.read(int(sys.argv[1]),1); '
            'os.close(int(sys.argv[1])); '
            'sys.exit(88) if ok != b"1" else os.execv(sys.executable,[sys.executable]+sys.argv[2:])')
    command = [sys.executable, '-c', gate, str(read_fd), driver, *overrides()]
    with log.open('x') as output:
        try:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, pass_fds=(read_fd,))
        except BaseException:
            os.close(write_fd)
            raise
        finally:
            os.close(read_fd)
        identity = {'pid': child.pid, 'create_time': None}
        identity_error = None
        try:
            identity['create_time'] = psutil.Process(child.pid).create_time()
            with identity_path.open('x') as stream:
                json.dump(dict(checked, **identity), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.write(write_fd, b'1')
        except Exception as exc:
            identity_error = repr(exc)
        finally:
            os.close(write_fd)
        rc = child.wait()
    with result.open('x') as stream:
        json.dump(dict(checked, **identity, actual_wait_returncode=rc, identity_error=identity_error,
                       finished_unix=time.time()), stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    if identity_error is not None:
        raise RuntimeError('driver identity recording failed after actual wait: ' + identity_error)
    return rc


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('preflight', 'run'):
        raise SystemExit('expected exactly preflight or run')
    if sys.argv[1] == 'preflight':
        print(json.dumps(preflight(os.environ), sort_keys=True))
    else:
        raise SystemExit(run(os.environ))
