"""Independent CPU-only loader regressions, never full-model GPU evidence."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import generate_hf_reference as hf

HAS_TORCH=all(importlib.util.find_spec(name) is not None for name in ('torch','accelerate','safetensors'))


class LoadPlanRegression(unittest.TestCase):
    def test_nonoverlapping_map_covers_all_60_layers(self):
        mapping=hf.reference_device_map(60)
        self.assertNotIn('',mapping)
        for layer in range(60):
            self.assertEqual(f'cuda:{layer//15}',hf.target_device(
                f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj',mapping))
        self.assertEqual('cuda:0',hf.target_device('model.language_model.rotary_emb.inv_freq',mapping))

    def test_root_or_other_ancestor_overlap_is_rejected_before_materialization(self):
        for mapping in ({'':'cuda:0','block':'cuda:1'},{'block':'cuda:0','block.child':'cuda:1'}):
            with self.assertRaises(ValueError):hf.validate_map(mapping)
        with self.assertRaises(ValueError):hf.target_device('missing.weight',{'block':'cuda:0'})

    def test_exact_bf16_and_scalar_fp32_bytes_with_16gib_headroom(self):
        shapes={'block.gate_up_proj':[512,2048,4096],'block.down_proj':[512,4096,1024],'block.A_log':[64]}
        plan=hf.parameter_plan(shapes,{'block.A_log'},{'block':'cuda:0'})
        self.assertEqual(12*1024**3+256,plan['total_parameter_bytes'])
        self.assertEqual(8*1024**3,plan['per_device']['cuda:0']['maximum_single_parameter_bytes'])
        self.assertEqual(16*1024**3,plan['minimum_free_after_load_bytes'])
        self.assertEqual(1,plan['maximum_live_cpu_source_tensors'])
        with self.assertRaises(ValueError):hf.parameter_plan(shapes,{'block.gate_up_proj'},{'block':'cuda:0'})

    def test_hf_gate_declaration_cannot_match_expert_gate_up_by_substring(self):
        model=types.SimpleNamespace(_keep_in_fp32_modules=[],_keep_in_fp32_modules_strict=['gate'],
                                    named_parameters=lambda:[('block.gate_up_proj',None),('block.A_log',None)])
        names,_=hf.declared_fp32_policy(model,{'block.A_log'})
        self.assertEqual({'block.A_log'},names)
        model._keep_in_fp32_modules_strict=['gate_up_proj']
        with self.assertRaises(RuntimeError):hf.declared_fp32_policy(model,{'block.A_log'})


@unittest.skipUnless(HAS_TORCH,'CPU Torch/Accelerate/safetensors runtime is not installed locally')
class ActualCpuMaterializationRegression(unittest.TestCase):
    def setUp(self):
        import torch
        from accelerate import init_empty_weights
        self.torch=torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block=torch.nn.Module()
                self.block.register_parameter('gate_up_proj',torch.nn.Parameter(torch.empty(2,4,3)))
                self.block.register_parameter('A_log',torch.nn.Parameter(torch.ones(2)))
                self.block.register_buffer('inv_freq',torch.tensor([1.,.5]))
                self.block.original_inv_freq=self.block.inv_freq
        with torch.device('cpu'),init_empty_weights(include_buffers=False):self.model=Tiny()
        self.assertTrue(all(p.is_meta for p in self.model.parameters()))

    def test_official_safetensor_style_assignment_explicit_dtypes_and_buffer_alias(self):
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from safetensors.torch import save_file
        torch=self.torch
        buffers,aliases=hf.buffer_inventory(self.model)
        values={'block.gate_up_proj':torch.arange(24).reshape(2,4,3).to(torch.bfloat16)/8,
                'block.A_log':torch.tensor([-1.,2.],dtype=torch.float32)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tiny-official-format.safetensors'
            save_file(values,path)
            with safe_open(path,framework='pt',device='cpu') as source:
                for name,expected in values.items():
                    dtype=torch.float32 if name.endswith('A_log') else torch.bfloat16
                    proof=hf.assign_parameter(self.model,name,source.get_tensor(name),list(expected.shape),'cpu',dtype,
                                              set_module_tensor_to_device)
                    self.assertTrue(proof['source_sample_match'])
                    self.assertEqual(dtype,self.model.get_parameter(name).dtype)
                    self.assertTrue(torch.equal(expected,self.model.get_parameter(name)))
        hf.place_buffers(self.model,buffers,aliases,{'block':'cpu'},set_module_tensor_to_device)
        self.assertTrue(torch.equal(torch.tensor([1.,.5]),self.model.block.inv_freq))
        self.assertIs(self.model.block.original_inv_freq,self.model.block.inv_freq)
        self.assertFalse(torch.cuda.is_initialized())

    def test_dispatch_guard_blocks_relocation_before_copy_and_restores_hook(self):
        from accelerate import hooks
        from accelerate.utils import set_module_tensor_to_device
        torch=self.torch
        for name,p in list(self.model.named_parameters()):
            dtype=torch.float32 if name.endswith('A_log') else torch.bfloat16
            hf.assign_parameter(self.model,name,torch.ones(tuple(p.shape)),list(p.shape),'cpu',dtype,set_module_tensor_to_device)
        before=hf.storage_inventory(self.model)
        original=hooks.set_module_tensor_to_device
        def stationary(model,**kwargs):
            hooks.set_module_tensor_to_device(model,'block.gate_up_proj','cpu')
            return model
        with patch('accelerate.dispatch_model',stationary):
            hf.guarded_dispatch(self.model,{'block':'cpu'})
        hf.require_storage_unchanged(before,self.model)
        self.assertIs(original,hooks.set_module_tensor_to_device)
        def moving(model,**kwargs):
            hooks.set_module_tensor_to_device(model,'block.gate_up_proj','meta')
            raise AssertionError('Relocation must have been rejected first')
        with patch('accelerate.dispatch_model',moving):
            with self.assertRaises(RuntimeError):hf.guarded_dispatch(self.model,{'block':'cpu'})
        self.assertIs(original,hooks.set_module_tensor_to_device)
        hf.require_storage_unchanged(before,self.model)
        self.model.block.gate_up_proj=torch.nn.Parameter(self.model.block.gate_up_proj.detach().clone())
        with self.assertRaises(RuntimeError):hf.require_storage_unchanged(before,self.model)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
