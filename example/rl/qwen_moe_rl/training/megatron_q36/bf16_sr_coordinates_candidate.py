"""Explicit logical BF16 stochastic-rounding coordinates, research only.

The caller must prove `canonical_name` identifies one logical tensor and bind
its actual layout from the model/loader. This module supports contiguous local
TP partitions obtained by Megatron's strided axis split; it does not infer a
logical expert ID, a QKV permutation, or a noncontiguous physical view.
"""
from dataclasses import dataclass
import math
import torch
from bf16_hybrid_adam_candidate import _signed64


@dataclass(frozen=True)
class SRCoordinates:
    canonical_name: str
    global_shape: tuple
    local_shape: tuple
    axis: int | None = None
    partition_count: int = 1
    partition_rank: int = 0
    partition_stride: int = 1

    def validate(self):
        if not isinstance(self.canonical_name,str) or not self.canonical_name:
            raise ValueError('explicit nonempty logical parameter name required')
        if (not isinstance(self.global_shape,tuple) or not isinstance(self.local_shape,tuple)
            or not self.global_shape or len(self.global_shape)!=len(self.local_shape)
            or any(type(x) is not int or x<1 for x in self.global_shape+self.local_shape)):
            raise ValueError('invalid positive global/local shapes')
        if any(type(x) is not int for x in (self.partition_count,self.partition_rank,self.partition_stride)):
            raise ValueError('integer partition metadata required')
        if self.partition_count<1 or not 0<=self.partition_rank<self.partition_count or self.partition_stride<1:
            raise ValueError('invalid partition metadata')
        if self.axis is None:
            if self.global_shape!=self.local_shape or (self.partition_count,self.partition_rank,self.partition_stride)!=(1,0,1):
                raise ValueError('replicas must use one common logical coordinate space')
            return self
        if type(self.axis) is not int or not 0<=self.axis<len(self.global_shape):
            raise ValueError('partition axis invalid')
        if self.partition_count==1 or self.local_shape[self.axis]%self.partition_stride:
            raise ValueError('partition count/stride invalid')
        expected=list(self.local_shape);expected[self.axis]*=self.partition_count
        if tuple(expected)!=self.global_shape:raise ValueError('partition global shape mismatch')
        return self

    def indices(self,start,stop,device='cpu'):
        self.validate();n=math.prod(self.local_shape)
        if type(start) is not int or type(stop) is not int or not 0<=start<=stop<=n:
            raise ValueError('chunk is outside local parameter')
        idx=torch.arange(start,stop,dtype=torch.int64,device=device)
        if self.axis is None:return idx
        inner=math.prod(self.local_shape[self.axis+1:]);local_axis=self.local_shape[self.axis]
        outer=idx//(local_axis*inner);axis=(idx//inner)%local_axis;tail=idx%inner
        block=local_axis//self.partition_stride
        global_axis=(axis//block)*(block*self.partition_count)+self.partition_rank*block+(axis%block)
        return (outer*self.global_shape[self.axis]+global_axis)*inner+tail


def random16_at(indices,seed):
    if indices.dtype!=torch.int64 or indices.ndim!=1:raise ValueError('int64 logical indices required')
    x=indices+_signed64(seed)+_signed64(0x9E3779B97F4A7C15)
    x=(x^((x>>30)&((1<<34)-1)))*_signed64(0xBF58476D1CE4E5B9)
    x=(x^((x>>27)&((1<<37)-1)))*_signed64(0x94D049BB133111EB)
    x=x^((x>>31)&((1<<33)-1))
    return (x>>48)&65535


def stochastic_round_coordinates(value,seed,coordinates,start):
    if value.dtype!=torch.float32 or not value.is_contiguous() or not torch.isfinite(value).all():
        raise ValueError('expected finite contiguous FP32 update chunk')
    coords=coordinates.indices(start,start+value.numel(),device=value.device)
    bits=value.view(torch.int32).reshape(-1).to(torch.int64)&0xffffffff
    result=(((bits>>16)+(random16_at(coords,seed)<(bits&65535)).long())&65535).to(torch.int16)
    result=result.view(torch.bfloat16).reshape(value.shape)
    if not torch.isfinite(result).all():raise FloatingPointError('BF16 stochastic rounding overflow')
    return result
