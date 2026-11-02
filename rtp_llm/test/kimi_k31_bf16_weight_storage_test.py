"""Shared KDA construction must release source projection storage after padding."""
import gc
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref
import torch
from rtp_llm.models_py.modules.kimi_k3 import attention as module
from rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion import prepare_bf16_projection
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.utils.model_weight import W

class StorageOwnerTest(unittest.TestCase):
    def test_shared_constructor_publishes_one_padded_owner(self):
        source=torch.arange(8*6284,dtype=torch.float32).reshape(8,6284).remainder(17).bfloat16()
        before=source.clone();reference=weakref.ref(source)
        weights={module.K3W.KDA_INPUT:source,W.linear_attn_f_b_w:torch.zeros(128,1536,dtype=torch.bfloat16),
                 W.linear_attn_out_w:torch.zeros(1536,8,dtype=torch.bfloat16),W.linear_attn_norm_w:torch.ones(128,dtype=torch.bfloat16)}
        del source
        parallel=NS(tp_size=8,role_type=module.RoleType.PREFILL)
        backend=NS(local_num_k_heads=12,local_num_v_heads=12,head_k_dim=128,head_v_dim=128,parallelism_config=parallel)
        config=NS(linear_attention_config=NS(),layernorm_eps=1e-5,
                  k3_runtime_config=NS(dsa=True,kda_use_full_rank_gate=True,kda_gate_lower_bound=None))
        with patch.object(module.LinearFactory,'create_linear',side_effect=lambda w,**kw:CudaF16Linear(w)), \
             patch.object(module,'KdaPagedPrefill',return_value=backend), \
             patch.object(module,'KdaDecode',return_value=NS()), \
             patch.object(module,'FusedRMSNormGated',return_value=NS()):
            attention=module.KimiK3KDA(config,parallel,weights)
        gc.collect();self.assertIsNone(reference())
        owner=weights[module.K3W.KDA_INPUT]
        self.assertEqual(tuple(owner.shape),(8,6400))
        self.assertEqual(owner.untyped_storage().data_ptr(),attention.input.weight.untyped_storage().data_ptr())
        torch.testing.assert_close(owner[:,:6284],before,rtol=0,atol=0)
        self.assertFalse(bool(owner[:,6284:].any()))

    def test_unpadded_prepare_keeps_existing_nn_storage(self):
        weight=torch.arange(128,dtype=torch.float32).reshape(8,16).bfloat16()
        projection=CudaF16Linear(weight)
        one=prepare_bf16_projection(projection);two=prepare_bf16_projection(projection)
        self.assertEqual(one.untyped_storage().data_ptr(),weight.untyped_storage().data_ptr())
        self.assertEqual(one.untyped_storage().data_ptr(),two.untyped_storage().data_ptr())

if __name__=='__main__':unittest.main()
