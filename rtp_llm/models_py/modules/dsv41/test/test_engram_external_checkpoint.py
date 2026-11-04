"""CPU contract tests for external checkpoint selection; no CUDA initialization."""
import importlib.util
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

spec=importlib.util.spec_from_file_location('engram_external_checkpoint_test',Path(__file__).resolve().parents[1]/'engram.py')
e=importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def config():
    return dict(engram_layer_ids=[1,14],engram_num_embeddings=[1000,1000],
                engram_max_ngram_size=4,engram_n_heads=2,engram_head_dim=32,
                engram_compressed_vocab_size=32,engram_pad_token_id=2,
                engram_vocab_size=13,rms_norm_eps=1e-20)


class Storage:
    def __init__(self):
        self.device=0
        self.ready=True
        self.uses=0
        tensors=[]
        for layer in (1,14):
            for role,width in [('weight',32),('scale',1)]:
                tensors.append(SimpleNamespace(name=f'layers.{layer}.engram.embed.{role}',shape=(1000,width)))
        self.validated=SimpleNamespace(manifest=SimpleNamespace(backend='fabric_egm',tensors=tensors))
    @contextmanager
    def use(self):
        if not self.ready: raise RuntimeError('storage is not READY')
        self.uses+=1
        try: yield
        finally: self.uses-=1


class Embedding:
    def __init__(self): self.releases=0
    def release_external_views(self): self.releases+=1


class Projection(torch.nn.Module):
    def __init__(self,weight,scale):
        super().__init__()
        self.weight=weight
        self.scale=scale


class ExternalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.prefix='layers.1.engram.'
        self.tensors={self.prefix+'wkv.weight':torch.ones(160,192).to(torch.float8_e4m3fn),
                      self.prefix+'wkv.scale':torch.full((5,6),127,dtype=torch.uint8),
                      self.prefix+'q_weight':torch.ones(4,32,dtype=torch.bfloat16),
                      self.prefix+'k_weight':torch.ones(4,32,dtype=torch.bfloat16)}
        save_file(self.tensors,self.root/'small.safetensors')
        index={name:'small.safetensors' for name in self.tensors}
        # These table files intentionally do not exist in the consumer checkpoint.
        index.update({self.prefix+'embed.weight':'absent-large.safetensors',
                      self.prefix+'embed.scale':'absent-large.safetensors'})
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
        self.storage=Storage()
        self.embedding=Embedding()

    @contextmanager
    def cpu_cuda_constructor(self):
        # Mock only CUDA placement/projection construction. safe_open and local
        # WKV/q/k reads are real; separate real-GPU integration tests cover compute.
        module=ModuleType('rtp_llm.models_py.modules.dsv41.utils')
        module.V41MXFP8Linear=Projection
        with patch.dict(sys.modules,{module.__name__:module}), \
             patch.object(torch.Tensor,'to',lambda tensor,*args,**kwargs:tensor), \
             patch.object(torch.cuda,'device',lambda device:nullcontext()), \
             patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CPU test initialized CUDA')), \
             patch.object(e.HostEngramEmbedding,'from_storage',return_value=self.embedding) as attach:
            yield attach

    def test_external_reads_only_small_weights_with_large_files_absent(self):
        import safetensors
        original=safetensors.safe_open
        opened=[]
        def tracked(path,*args,**kwargs):
            opened.append(Path(path).name)
            return original(path,*args,**kwargs)
        with self.cpu_cuda_constructor() as attach, patch.object(safetensors,'safe_open',side_effect=tracked):
            model=e.Engram.from_checkpoint(config(),1,str(self.root),'cuda:0',external_storage=self.storage)
        attach.assert_called_once_with(self.storage,1)
        self.assertEqual(opened,['small.safetensors']*4)
        self.assertIs(model.embed_tokens,self.embedding)
        self.assertEqual(self.embedding.releases,0)
        self.assertEqual(self.storage.uses,0)
        self.assertEqual(tuple(model.q_weight.shape),(4,32))
        model.release_external_views()
        self.assertEqual(self.embedding.releases,1)

    def test_projection_or_qk_failure_releases_external_views(self):
        index=json.loads((self.root/'model.safetensors.index.json').read_text())
        del index['weight_map'][self.prefix+'k_weight']
        (self.root/'model.safetensors.index.json').write_text(json.dumps(index))
        with self.cpu_cuda_constructor(), self.assertRaises(KeyError):
            e.Engram.from_checkpoint(config(),1,str(self.root),'cuda:0',external_storage=self.storage)
        self.assertEqual(self.embedding.releases,1)

    def test_external_device_mismatch_fails_before_checkpoint_io(self):
        for device in ['cpu','cuda:1']:
            with self.subTest(device=device), self.assertRaises(ValueError):
                e.Engram.from_checkpoint(config(),1,'/does-not-exist',device,external_storage=self.storage)

    def test_complete_layout_checked_before_opening_checkpoint(self):
        # A valid requested layer cannot hide a missing or mismatched other layer.
        mutations=[lambda ts:ts[:-1],lambda ts:ts+[SimpleNamespace(name='layers.99.engram.embed.weight',shape=(1,32))],
                   lambda ts:[SimpleNamespace(name=t.name,shape=(7,1)) if t.name.endswith('14.engram.embed.scale') else t for t in ts]]
        for mutation in mutations:
            storage=Storage()
            storage.validated.manifest.tensors=mutation(storage.validated.manifest.tensors)
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'complete model'):
                e.Engram.from_checkpoint(config(),1,'/does-not-exist','cuda:0',external_storage=storage)

    def test_nonready_storage_fails_before_checkpoint_io(self):
        self.storage.ready=False
        with self.assertRaisesRegex(RuntimeError,'READY'):
            e.Engram.from_checkpoint(config(),1,'/does-not-exist','cuda:0',external_storage=self.storage)

    def test_wrong_backend_never_falls_back_to_legacy(self):
        self.storage.validated.manifest.backend='local_shm'
        with self.assertRaisesRegex(ValueError,'complete model'):
            e.Engram.from_checkpoint(config(),1,'/does-not-exist','cuda:0',external_storage=self.storage)

    def test_legacy_still_requires_real_table_files(self):
        with self.assertRaises(FileNotFoundError):
            e.Engram.from_checkpoint(config(),1,str(self.root),'cpu')

    def test_table_shape_is_metadata_for_both_backends(self):
        legacy=e.HostEngramEmbedding(torch.zeros(3,32,dtype=torch.float8_e4m3fn),torch.full((3,1),127,dtype=torch.uint8))
        self.assertEqual(legacy.table_shape,(3,32))
        external=e.HostEngramEmbedding.__new__(e.HostEngramEmbedding)
        external._uses_external_storage=True
        external._table_shape=(1000,32)
        external.weight=None
        self.assertEqual(external.table_shape,(1000,32))

    def external_embedding(self):
        embedding=e.HostEngramEmbedding.__new__(e.HostEngramEmbedding)
        embedding._uses_external_storage=True
        embedding.weight=embedding.scales=None
        embedding._pinned=None
        return embedding

    def lookup_model(self, embedding):
        return SimpleNamespace(
            _is_decode_role=False, _engram_cpu_runtime=None,
            _engram_layers=((1,0),(14,1)),
            v4=SimpleNamespace(layers={i:SimpleNamespace(
                engram=SimpleNamespace(embed_tokens=embedding)
            ) for i in (1,14)}),
        )

    def test_cpu_and_prefetch_capabilities_distinguish_backends(self):
        local=e.HostEngramEmbedding(torch.zeros(3,32,dtype=torch.uint8),torch.ones(3,1,dtype=torch.uint8))
        external=self.external_embedding()
        self.assertTrue(local.supports_cpu_lookup)
        self.assertTrue(local.supports_async_prefetch)
        self.assertFalse(external.supports_cpu_lookup)
        self.assertFalse(external.supports_async_prefetch)

    def test_local_default_cpu_lookup_still_prepares_once(self):
        local=e.HostEngramEmbedding(torch.zeros(3,32,dtype=torch.uint8),torch.ones(3,1,dtype=torch.uint8))
        model=self.lookup_model(local)
        runtime=SimpleNamespace(capacity=16,slots={})
        with patch.dict('os.environ',{},clear=True), \
             patch.object(e,'EngramCPULookupRuntime',return_value=runtime) as create:
            self.assertTrue(e.engram_cpu_lookup_enabled())
            self.assertFalse(e.engram_cpu_lookup_enabled(is_decode_role=True))
            e.prepare_engram_cpu_lookup(model,max_m=16)
            e.prepare_engram_cpu_lookup(model,max_m=16)
        create.assert_called_once_with(model,16)
        self.assertIs(model._engram_cpu_runtime,runtime)

    def test_external_cpu_flag_never_creates_runtime_or_reads_missing_host_views(self):
        for env in ({},{'DSV41_CPU_ENGRAM_LOOKUP':'1'},
                    {'DSV41_CPU_ENGRAM_LOOKUP':'1','DSV41_ASYNC_ENGRAM_LOOKUP':'1'}):
            model=self.lookup_model(self.external_embedding())
            with self.subTest(env=env), patch.dict('os.environ',env,clear=True), \
                 patch.object(e,'EngramCPULookupRuntime',side_effect=AssertionError('CPU runtime for external views')) as create:
                e.prepare_engram_cpu_lookup(model,max_m=16)
                e.prepare_engram_cpu_lookup(model,max_m=16)
                create.assert_not_called()
                self.assertIsNone(model._engram_cpu_runtime)

    def test_mixed_model_does_not_submit_cpu_work_for_gpu_only_layer(self):
        local=e.HostEngramEmbedding(torch.zeros(3,32,dtype=torch.uint8),torch.ones(3,1,dtype=torch.uint8))
        model=self.lookup_model(local)
        model.v4.layers[14].engram.embed_tokens=self.external_embedding()
        with patch.dict('os.environ',{'DSV41_CPU_ENGRAM_LOOKUP':'1'}), \
             patch.object(e,'EngramCPULookupRuntime') as create:
            e.prepare_engram_cpu_lookup(model,max_m=16)
        create.assert_not_called()

    def test_external_async_flag_creates_no_stream_event_or_lookup(self):
        external=self.external_embedding()
        # Missing CUDA metadata deliberately makes any access beyond the
        # capability check fail, without initializing a CUDA context.
        module=SimpleNamespace(embed_tokens=external,_lookup_stream=None)
        with patch.dict('os.environ',{'DSV41_CPU_ENGRAM_LOOKUP':'0','DSV41_ASYNC_ENGRAM_LOOKUP':'1'}), \
             patch.object(torch.cuda,'Stream',side_effect=AssertionError('stream created')) as stream, \
             patch.object(torch.cuda,'Event',side_effect=AssertionError('event created')) as event:
            e.Engram.prepare_lookup_prefetch(module)
            self.assertIsNone(e.Engram.prefetch_lookup(module,object()))
            self.assertIsNone(external.warmup_prefetch(object(),'cuda:0'))
            with self.assertRaisesRegex(RuntimeError,'asynchronous prefetch is not supported'):
                external(object(),'cuda:0',prefetch=True)
        stream.assert_not_called()
        event.assert_not_called()


if __name__=='__main__': unittest.main()
