
import functools
import logging
import torch

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, concat_1, concat_0, identity, zeros, transpose, sp_id
from maga_transformer.models.qwen import QWenWeight

class ClipQWenVLVitWeights:
    vit_conv1 = 'transformer.visual.conv1.weight'
    vit_ln_pre_w = 'transformer.visual.ln_pre.weight'
    vit_ln_pre_b = 'transformer.visual.ln_pre.bias'
    vit_ln_post_w = 'transformer.visual.ln_post.weight'
    vit_ln_post_b = 'transformer.visual.ln_post.bias'
    vit_proj = 'transformer.visual.proj'
    vit_positional_embedding = 'transformer.visual.positional_embedding'

    vit_attn_block_ln_1_w = 'transformer.visual.transformer.resblocks.{i}.ln_1.weight'
    vit_attn_block_ln_1_b = 'transformer.visual.transformer.resblocks.{i}.ln_1.bias'
    vit_attn_block_ln_2_w = 'transformer.visual.transformer.resblocks.{i}.ln_2.weight'
    vit_attn_block_ln_2_b = 'transformer.visual.transformer.resblocks.{i}.ln_2.bias'
    vit_attn_block_attn_in_proj_w = 'transformer.visual.transformer.resblocks.{i}.attn.in_proj.weight'
    vit_attn_block_attn_in_proj_b = 'transformer.visual.transformer.resblocks.{i}.attn.in_proj.bias'
    vit_attn_block_attn_out_proj_w = 'transformer.visual.transformer.resblocks.{i}.attn.out_proj.weight'
    vit_attn_block_attn_out_proj_b = 'transformer.visual.transformer.resblocks.{i}.attn.out_proj.bias'
    vit_attn_block_mlp_c_fc_w = 'transformer.visual.transformer.resblocks.{i}.mlp.c_fc.weight'
    vit_attn_block_mlp_c_fc_b = 'transformer.visual.transformer.resblocks.{i}.mlp.c_fc.bias'
    vit_attn_block_mlp_c_proj_w = 'transformer.visual.transformer.resblocks.{i}.mlp.c_proj.weight'
    vit_attn_block_mlp_c_proj_b = 'transformer.visual.transformer.resblocks.{i}.mlp.c_proj.bias'
    
    vit_attn_pool_kv_proj_w = 'transformer.visual.attn_pool.kv_proj.weight'
    vit_attn_pool_attn_in_proj_w = 'transformer.visual.attn_pool.attn.in_proj_weight'
    vit_attn_pool_attn_in_proj_b = 'transformer.visual.attn_pool.attn.in_proj_bias'
    vit_attn_pool_attn_out_proj_w = 'transformer.visual.attn_pool.attn.out_proj.weight'
    vit_attn_pool_attn_out_proj_b = 'transformer.visual.attn_pool.attn.out_proj.bias'
    vit_attn_pool_ln_q_w = 'transformer.visual.attn_pool.ln_q.weight'
    vit_attn_pool_ln_q_b = 'transformer.visual.attn_pool.ln_q.bias'
    vit_attn_pool_ln_kv_w = 'transformer.visual.attn_pool.ln_kv.weight'
    vit_attn_pool_ln_kv_b = 'transformer.visual.attn_pool.ln_kv.bias'
    vit_attn_pool_pos_emb = 'transformer.visual.attn_pool.pos_embed'
    vit_attn_pool_query = 'transformer.visual.attn_pool.query'

    vit_weights = {
        vit_conv1,
        vit_ln_pre_w,
        vit_ln_pre_b,
        vit_ln_post_w,
        vit_ln_post_b,
        vit_proj,
        vit_positional_embedding
    }

    vit_layer_weights = {
        vit_attn_block_ln_1_w,
        vit_attn_block_ln_1_b,
        vit_attn_block_ln_2_w,
        vit_attn_block_ln_2_b,
        vit_attn_block_attn_in_proj_w,
        vit_attn_block_attn_in_proj_b,
        vit_attn_block_attn_out_proj_w,
        vit_attn_block_attn_out_proj_b,
        vit_attn_block_mlp_c_fc_w,
        vit_attn_block_mlp_c_fc_b,
        vit_attn_block_mlp_c_proj_w,
        vit_attn_block_mlp_c_proj_b
    }

    vit_attn_pool_weight = {
        vit_attn_pool_kv_proj_w,
        vit_attn_pool_attn_in_proj_w,
        vit_attn_pool_attn_in_proj_b,
        vit_attn_pool_attn_out_proj_w,
        vit_attn_pool_attn_out_proj_b,
        vit_attn_pool_ln_q_w,
        vit_attn_pool_ln_q_b,
        vit_attn_pool_ln_kv_w,
        vit_attn_pool_ln_kv_b,
        vit_attn_pool_pos_emb,
        vit_attn_pool_query
    }

class QWenVLWeightInfo(QWenWeight):
    def __init__(self, config, tp_size, tp_rank):
        super().__init__(config, tp_size, tp_rank)
        self.layers = config.vit_related_params.layers
    
    def _get_weight_info(self):
        qwen_vl_weight = super()._get_weight_info()

        for weight in set.union(ClipQWenVLVitWeights.vit_weights, ClipQWenVLVitWeights.vit_attn_pool_weight):
            qwen_vl_weight.weights.append(WeightInfo(weight, [CkptWeightInfo(weight, identity)], identity))
            qwen_vl_weight.tp_strategy[weight] = sp_id

        for i in range(self.layers):
            for weight in ClipQWenVLVitWeights.vit_layer_weights:
                w = weight.format(i = i)
                qwen_vl_weight.weights.append(WeightInfo(w, [CkptWeightInfo(w, identity)], identity))
                qwen_vl_weight.tp_strategy[w] = sp_id

        return qwen_vl_weight