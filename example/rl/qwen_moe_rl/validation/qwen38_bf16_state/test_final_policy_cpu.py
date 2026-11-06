"""CPU fixtures test control gates; these are not native/RL acceptance."""
import asyncio
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import qwen38_final_policy as module


def proof_fixture():
    scores=[-1.5,-3.25]
    return dict(final_native_GPU_reprobe=True,global_step=2,logical_model_replicas=1,
        world_size=72,tp_size=8,dp_size=9,ep_size=72,model_version='test-only-v2',native_incarnation='test-only',
        full_source_sha256='1'*64,prompt_ids=[4,6],response_ids=[7,8],hf_raw_log_probs=scores,max_HF_error=0.,
        statuses=[dict(world_rank=r,global_step=2,model_version='test-only-v2',phase='running',
            received_weights=1590,applied_weights=1,update_id=None) for r in range(72)],
        probes=[dict(dp_rank=dp,generation_world_rank=dp*8,raw_log_probs=scores,
            raw_fp32_bits_hex=struct.pack('<2f',*scores).hex()) for dp in range(9)])


async def main():
    fixture=proof_fixture();module.validate_final_policy_proof(fixture)
    rejected=0
    for mutation in [lambda p:p.update(global_step=1),lambda p:p.update(final_native_GPU_reprobe=False),
        lambda p:p['statuses'][7].update(received_weights=1589),lambda p:p['statuses'][8].update(world_rank=7),
        lambda p:p['probes'][1].update(dp_rank=0),lambda p:p.update(max_HF_error=.1),
        lambda p:p['probes'][0].update(raw_fp32_bits_hex='00')]:
        changed=copy.deepcopy(fixture);mutation(changed)
        try:module.validate_final_policy_proof(changed)
        except RuntimeError:rejected+=1
        else:raise AssertionError('An incomplete/stale/forged fixture must fail the gate')
    os.environ.update(Q38_ENABLE_BF16_STATE_RUNTIME='1',RTP_LLM_RL_WORLD72='1')
    with tempfile.TemporaryDirectory() as temp:
        os.environ['Q38_RUN_ROOT']=temp
        module.install_backend_hooks()
        from rllm.trainer.verl.verl_backend import VerlBackend
        from omegaconf import OmegaConf
        import torch
        backend=object.__new__(VerlBackend)
        backend.config=OmegaConf.create(dict(trainer=dict(save_freq=-1),
            actor_rollout_ref=dict(rollout=dict(name='rtp'))))
        backend.is_separated=False
        syncs=[];native_calls=[]
        async def sync(step):syncs.append(step)
        async def remote(step):native_calls.append(step);return proof_fixture()
        backend.checkpoint_manager=SimpleNamespace(update_weights=sync)
        replica=SimpleNamespace(replica_rank=0,world_size=72,workers=list(range(72)),
            _server_handle=SimpleNamespace(verify_final_policy=SimpleNamespace(remote=remote)))
        backend.llm_server_manager=SimpleNamespace(get_replicas=lambda:[replica])
        for step in (1,2):
            state=SimpleNamespace(global_step=step,is_training=True,timing_dict={},has_backend_batch=False)
            await backend.on_batch_end(state)
        await backend.on_train_end(state)
        module.require_final_policy_completion(backend)
        assert syncs==[1,2] and native_calls==[2]
        assert len(list(Path(temp).glob('final-policy2-verified-trainer-*.json')))==1
        assert not torch.cuda.is_initialized()
        del backend._q38_final_policy_proof
        try:module.require_final_policy_completion(backend)
        except RuntimeError:pass
        else:raise AssertionError('A swallowed final-hook failure must fail completion')
        # Exercise the actual restored fit_async finally block: it swallows a
        # failing on_train_end RPC, and our outer completion gate must rethrow.
        from rllm.trainer.unified_trainer import UnifiedTrainer
        from unittest.mock import patch
        trainer=object.__new__(UnifiedTrainer)
        trainer.config=backend.config;trainer.backend=backend;trainer._remote_runtime=None
        trainer.agent_workflow_engine=SimpleNamespace();trainer._train_dataloader=None;trainer._gateway=None
        trainer.rllm_config=OmegaConf.create(dict(trainer=dict(val_before_train=False)))
        async def train_start(state):pass
        async def tiny_fit(state):state.global_step=2
        async def failed_remote(step):raise RuntimeError('injected final native RPC failure; CPU fixture')
        backend.on_train_start=train_start;trainer._fit_async=tiny_fit
        replica._server_handle.verify_final_policy.remote=failed_remote
        actual_old_fit=UnifiedTrainer.fit_async.__wrapped__
        with patch.dict(actual_old_fit.__globals__,{'print_config_table':lambda *a,**k:None}):
            try:await trainer.fit_async()
            except RuntimeError as error:
                assert 'without real final policy-2' in str(error)
            else:raise AssertionError('Actual rLLM swallowed final RPC failure and escaped the completion gate')
    # Passive setup is separately exercised in a fresh process with a bridge
    # fixture; importing it must not import either Torch or Ray.
    code='''import sys,types
bridge=types.ModuleType('install_rtp_bridge')
bridge._install_driver=lambda:None
bridge.worker_setup=lambda:None
sys.modules['install_rtp_bridge']=bridge
import qwen38_final_policy
qwen38_final_policy.worker_setup()
assert 'torch' not in sys.modules and 'ray' not in sys.modules
assert bridge._install_driver._q38_final_policy_driver
'''
    subprocess.run([sys.executable,'-c',code],check=True)
    result=dict(schema='q38-final-policy-control-CPU-fixtures-v1',passed=True,
        invalid_proofs_rejected=rejected,real_restored_VerBackend_on_batch_end_executed=True,
        fixture_checkpoint_sync_steps=syncs,fixture_final_RPC_steps=native_calls,
        third_optimizer_or_duplicate_checkpoint_sync_invoked=False,
        actual_restored_fit_finally_swallow_cannot_hide_failed_native_RPC=True,
        passive_setup_did_not_import_Torch_or_Ray=True,actual_native_GPU_proof=False)
    Path('final-policy-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':asyncio.run(main())
