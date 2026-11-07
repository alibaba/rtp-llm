"""Actual bounded loader copy/empty shard contract, tiny CPU tensors only."""
import argparse,hashlib,json,os
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import qwen38_bf16_streamed_checkpoint as loader


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
 os.environ['Q38_REPLAY_VERIFY_FULL_SHARD_BYTES']='1'
 source=torch.arange(24,dtype=torch.float32).reshape(3,8).to(torch.bfloat16)
 class Slice:
  def get_shape(self):return list(source.shape)
  def __getitem__(self,selection):return source[selection]
 class Parameter:
  def __init__(self,tensor):self.tensor=tensor
  def detach(self):return self
  def to_local(self):return self.tensor
 destination=torch.empty_like(source);parameter=Parameter(destination)
 with patch.object(loader,'source_interval',return_value=(0,3)):
  row=loader.copy_source_slice(parameter,Slice(),max_source_bytes=16,world_size=1)
 assert torch.equal(destination.view(torch.uint8),source.view(torch.uint8))
 assert row['full_destination_BF16_bytes_verified'] and row['source_reads']==3
 expected=[hashlib.sha256(memoryview(source[i:i+1].view(torch.uint8).numpy()).cast('B')).hexdigest() for i in range(3)]
 assert row['full_slice_bytes_sha256']==expected
 with patch.object(loader,'source_interval',return_value=(3,3)):
  empty=loader.copy_source_slice(Parameter(torch.empty(0,8,dtype=torch.bfloat16)),Slice(),max_source_bytes=16,world_size=72)
 assert empty['local_numel']==0 and empty['source_reads']==0
 assert empty['full_destination_BF16_bytes_verified'] and empty['full_slice_bytes_sha256']==[]
 for value in [source.float(),source[:,:4]]:
  try:loader._verify_copy(destination,value)
  except ValueError:pass
  else:raise AssertionError('Wrong source dtype/shape accepted')
 os.environ.pop('Q38_REPLAY_VERIFY_FULL_SHARD_BYTES')
 historical=loader._verify_copy(destination,source)
 assert isinstance(historical,str) and len(historical)==64
 assert not torch.cuda.is_initialized()
 record=dict(schema='q38-bounded-current-full-byte-copy-CPU-v1',passed=True,
  real_loader_functions=True,all_current_BF16_bytes_equal=True,three_bounded_fragments_verified=True,
  zero_local_shard_vacuous_full_coverage_checked=True,historical_sample_mode_unchanged=True,
  wrong_dtype_and_shape_rejected=True,CUDA_initialized=False,official_model_weights_opened=False)
 Path(args.output).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
