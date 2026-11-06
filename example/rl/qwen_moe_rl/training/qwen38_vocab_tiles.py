"""Meta-only vocabulary row subdivision with unchanged logits and embeddings."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class VocabTile(nn.Module):
    def __init__(self,weight,row_begin,*,padding_idx=None,copy_values=False):
        super().__init__()
        self.weight=nn.Parameter(weight.detach().clone() if copy_values else weight)
        self.row_begin=row_begin;self.row_end=row_begin+weight.shape[0]
        self.padding_idx=padding_idx-self.row_begin if (padding_idx is not None
                                    and self.row_begin<=padding_idx<self.row_end) else None

    def forward(self,value,*,embedding=False):
        if not embedding:return F.linear(value,self.weight)
        selected=(value>=self.row_begin)&(value<self.row_end)
        indices=(value-self.row_begin).clamp(0,self.weight.shape[0]-1)
        result=F.embedding(indices,self.weight,padding_idx=self.padding_idx)
        return result*selected.unsqueeze(-1)


class TiledEmbedding(nn.Module):
    def __init__(self,original,*,rows=16384,require_meta=True):
        super().__init__()
        if not isinstance(original,nn.Embedding):raise TypeError('Expected the original HF embedding module')
        if require_meta and not original.weight.is_meta:raise ValueError('Do not copy a materialized full vocabulary')
        self.num_embeddings=original.num_embeddings;self.embedding_dim=original.embedding_dim
        self.padding_idx=original.padding_idx
        self.tiles=nn.ModuleList(VocabTile(original.weight[begin:begin+rows],begin,
                              padding_idx=self.padding_idx,copy_values=not require_meta)
                                for begin in range(0,self.num_embeddings,rows))

    def forward(self,input_ids):
        result=None
        for tile in self.tiles:
            value=tile(input_ids,embedding=True)
            result=value if result is None else result+value
        return result


class TiledLMHead(nn.Module):
    def __init__(self,original,*,rows=16384,require_meta=True):
        super().__init__()
        if not isinstance(original,nn.Linear) or original.bias is not None:
            raise TypeError('The pinned untied biasless HF linear vocabulary head is required')
        if require_meta and not original.weight.is_meta:raise ValueError('Do not copy a materialized full vocabulary')
        self.in_features=original.in_features;self.out_features=original.out_features
        self.tiles=nn.ModuleList(VocabTile(original.weight[begin:begin+rows],begin,
                                         copy_values=not require_meta)
                                for begin in range(0,self.out_features,rows))

    def forward(self,hidden_states):
        return torch.cat([tile(hidden_states) for tile in self.tiles],dim=-1)


def install_meta_vocab_tiles(model,*,rows=16384):
    if rows!=16384:raise ValueError('Initial production profile fixes the vocabulary wrap size to 16,384 rows')
    if model.config.vocab_size!=248320 or model.config.hidden_size!=8192 or model.config.tie_word_embeddings:
        raise ValueError('Only the pinned independent input/output vocabulary matrices are supported')
    sources=model._qwen38_expert_sources
    for prefix,old,new in [('model.embed_tokens',model.model.embed_tokens,TiledEmbedding),
                           ('lm_head',model.lm_head,TiledLMHead)]:
        replacement=new(old,rows=rows,require_meta=True)
        parent,attribute=prefix.rsplit('.',1) if '.' in prefix else ('',prefix)
        setattr(model.get_submodule(parent),attribute,replacement)
        for index,tile in enumerate(replacement.tiles):
            sources[f'{prefix}.tiles.{index}.weight']={
                'canonical_name':prefix+'.weight','runtime_shape':list(tile.weight.shape),
                'source_shape':list(old.weight.shape),'vocab_row_begin':tile.row_begin,
                'vocab_row_end':tile.row_end}
    return sources


def shard_vocab_tiles_before_root_wrap(model,fsdp_kwargs):
    from torch.distributed.fsdp import fully_shard
    count=0
    for tile in model.modules():
        if isinstance(tile,VocabTile):fully_shard(tile,**fsdp_kwargs);count+=1
    if count!=32:raise ValueError('Expected sixteen input and sixteen output vocabulary tiles')
    return count
