"""Eight-rank CPU protocol checks; this is not a GPU/model acceptance test."""
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class TensorEnvelope:
    """CPU payload with a fake transport device for protocol logic only."""
    def __init__(self,value):self.value=value
    @property
    def dtype(self):return self.value.dtype
    @property
    def shape(self):return self.value.shape
    @property
    def is_cuda(self):return True
    def detach(self):return self
    def to(self,dtype=None):return TensorEnvelope(self.value.to(dtype))
    def contiguous(self):return TensorEnvelope(self.value.contiguous())
    def cpu(self):return self.value


def worker(rank,world,port,output):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),WORLD_SIZE=str(world),
                     RLLM_RTP_ACCEPTANCE_DIR=output)
    dist.init_process_group('gloo',rank=rank,world_size=world)
    import rtp_verl_backend as mod
    from rtp_topology import ReplicaTopology, training_rank_layout
    class FakeIPC:
        @staticmethod
        def export_tensor(value):return bytes([rank])
    sys.modules['rtp_llm.utils.cuda_ipc']=types.SimpleNamespace(CudaIpcV2=FakeIPC)
    mod._source=lambda:None
    reports=[]
    for tp in [8,2]:
        groups=[dist.new_group(list(range(start,start+tp)),backend='gloo') for start in range(0,world,tp)]
        layout=training_rank_layout(rank,world,tp)
        obj=mod.RTPServerAdapter.__new__(mod.RTPServerAdapter)
        for k,v in layout.items():setattr(obj,k,v)
        obj.distributed=True;obj.global_rank=rank;obj.tp_group=groups[rank//tp]
        obj.is_leader=obj.tp_rank==0;obj._unacknowledged_ipc=[];obj.tied_embeddings=False
        names=['model.language_model.embed_tokens.weight','model.language_model.layers.0.linear_attn.A_log',
               'model.language_model.layers.0.linear_attn.dt_bias']
        obj.hf_all_sources=set(names)|{'model.visual.weight'}
        obj._required_sources=lambda:set(names)
        def global_failure(error):
            flag=torch.tensor(int(error is not None));dist.all_reduce(flag,op=dist.ReduceOp.MAX);return bool(flag.item())
        obj._any_global_failure=global_failure
        counters={'sleep':0,'push':0,'terminated':0}
        failure_mode=['none']
        async def sleep(replica):counters['sleep']+=1
        mod._sleep=sleep
        asyncio.run(obj.release())
        total=torch.tensor(counters['sleep']);dist.all_reduce(total)
        assert int(total)==world//tp,(tp,total)
        peers=obj._gather({'rank':rank})
        assert [x['rank'] for x in peers]==layout['group_ranks']
        async def terminate():counters['terminated']+=1;obj._unacknowledged_ipc=[]
        obj._terminate_owned_replica=terminate
        class FakeControl:
            def __init__(self):self.step=None;self.received=[]
            async def status(self):return [types.SimpleNamespace(model_version='initial',phase='paused')]*tp
            async def begin_update(self,global_step,max_buffer_bytes):self.step=global_step
            async def push_weight(self,name,metadata):
                assert metadata==[bytes([i]) for i in layout['group_ranks']]
                if failure_mode[0]=='ack' and layout['replica_rank']==0 and name.endswith('.A_log'):
                    raise RuntimeError('injected native WRITE_WEIGHT ACK failure')
                self.received.append(name);counters['push']+=1
            async def commit_update(self):
                assert self.received==names
                return [types.SimpleNamespace(model_version='updated',global_step=self.step,received_weights=3,
                    applied_weights=3,tp_rank=i,tp_size=tp,phase='paused') for i in range(tp)]
            async def close(self):pass
        async def control(replica):return FakeControl()
        mod._new_control=control
        consumed=[]
        def stream(fail=False):
            for index,name in enumerate(names+['model.visual.weight']):
                # Mimic FSDP full_tensor's global collective for every generator item.
                value=torch.tensor(index);dist.all_reduce(value)
                assert int(value)==index*world
                if failure_mode[0]=='name_order' and rank==3 and index==0:
                    name='model.visual.weight'
                consumed.append(name)
                dtype=torch.bfloat16 if fail and rank==3 and name.endswith('.A_log') else torch.float32
                yield name,TensorEnvelope(torch.arange(4,dtype=dtype))
        asyncio.run(obj.update_weights(stream(),global_steps=1))
        assert len(consumed)==4
        assert counters['push']==(3 if obj.is_leader else 0)
        assert not obj._unacknowledged_ipc
        before=len(consumed)
        try:asyncio.run(obj.update_weights(stream(fail=True),global_steps=2))
        except RuntimeError as e:assert 'synchronization failed' in str(e)
        else:raise AssertionError('one-rank FP32 source downcast did not fail the full training world')
        assert len(consumed)-before==4 and counters['terminated']==1
        failure_mode[0]='ack';before=len(consumed)
        try:asyncio.run(obj.update_weights(stream(),global_steps=2))
        except RuntimeError as e:assert 'synchronization failed' in str(e)
        else:raise AssertionError('native ACK failure did not fail the training world')
        assert len(consumed)-before==4 and counters['terminated']==2
        failure_mode[0]='name_order';before=len(consumed)
        try:asyncio.run(obj.update_weights(stream(),global_steps=2))
        except RuntimeError as e:assert 'synchronization failed' in str(e)
        else:raise AssertionError('one-rank filtered-source name mismatch did not fail the world')
        assert len(consumed)-before==4 and counters['terminated']==3
        reports.append(dict(tp_size=tp,replica_count=world//tp,source_rank_order=layout['group_ranks'],
                            full_iterator_consumed_after_peer_failure=True,ack_failure_fails_all_replicas=True,
                            filtered_source_name_mismatch_fails_without_deadlock=True,
                            leader_only_rpc=True))
    all_reports=[None]*world;dist.all_gather_object(all_reports,reports)
    if rank==0:
        assert not torch.cuda.is_initialized()
        Path(output,'cpu-distributed-protocol.json').write_text(json.dumps(dict(passed=True,world_size=world,
            device='cpu',cuda_used=False,reports=all_reports),indent=2)+'\n')
        print('CPU_DISTRIBUTED_PROTOCOL_PASSED',world,flush=True)
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(8,port,str(args.output)),nprocs=8,join=True)


if __name__=='__main__':main()
