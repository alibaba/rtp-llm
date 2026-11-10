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
  'run_standalone_q36_candidate_v3.py': '6378b814fed623fde292e7552165aecce1aa13893c7305e7e3bdbba5e54b20a1',
  'onpolicy_standalone_candidate.py': '05b8eb35666a68dc031aa465a0e7fa90cf01ceca407746763f60b63c75dbf15d',
  'q36_4plus4_claim_loader.py': '0075f76555d41822bfd23d3879165aa754b0b0f19484f99ec3f5166dfbe0c5ac',
  'q36_4plus4_claim_guard.py': '8bbfcd82128c1c2ac4018ec1f9fd101547389523077acbbd4c3435c5321f9e0e',
  'q36_standalone_bootstrap_selector.py': '76917010ec8cf2782b4f0323b19811d109f48d27daabf5645be758fe2e117d9f',
  'standalone_bootstrap_v4.py': 'd3c993deb80505b578b97fd5222ff396dec4f0b5362dc6937aa503ff211e66a8'},
 'Q36_ADAPTER_DIR': {
  'standalone_rtp_adapter_v3.py': 'b849f17ab606e0554db3802eb43979dd10cedd64361cc8082f716c173cf64f5c',
  'standalone_rtp_adapter_v4.py': '0e455e9e8e0b363c09cda22bcd674cdb4775784463e798c595c6c07b8e0d5e83',
  'rtp_cleanup.py': '4506fe79022d0d02f14118ee30d65f1a935a4bb63e51d538831287fb663bdb5b'},
 'Q36_DISAGG_CODE_DIR': {
  'verl_nixl_guarded.py': '72716bd1e687b8a005cf0400a7ec998c5e440937268a301c0cc47b47cba3f1c8',
  'full_hf_relay.py': '0043b7fee1b6e80551174460cb1401aa2eb6c4216f346778c202984ab1fe55af'}}
MODEL_SHA = {'config.json': '93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99',
             'model.safetensors.index.json': '41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83'}
REQUIRED_FLAGS = {'Q36_STRICT_4PLUS4_CLAIM':'1', 'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1',
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
