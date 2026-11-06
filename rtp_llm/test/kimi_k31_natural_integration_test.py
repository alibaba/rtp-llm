"""CPU contracts for natural producers and complete Zigzag query execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch
import sys
import unittest

import torch

from rtp_llm.test.kimi_k31_natural_outer_layout_test import fixture, load_hybrid


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dsa():
    """Replace import-only native bindings; tensor operations remain real Torch."""
    modules = {}
    def stub(name, **values):
        mod = ModuleType(name)
        mod.__dict__.update(values)
        modules[name] = mod
    stub("rtp_llm.models_py.distributed.collective_torch", Group=SimpleNamespace(TP=0),
         all_gather=None, reduce_scatter_bf16=None)
    stub("rtp_llm.models_py.modules.hybrid.indexer", Indexer=None)
    stub("rtp_llm.models_py.modules.kimi_k3.attention", KimiK3MLA=torch.nn.Module)
    stub("rtp_llm.models_py.triton_kernels.common.activation",
         gate_sigmoid_mul=lambda x, gate, mask: x * gate.sigmoid() if mask is None else
         (x * gate.sigmoid()).masked_fill(~mask[:, None], 0))
    def norm(x, weight, eps, **kwargs):
        out = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
               * weight.float()).to(x.dtype)
        suffix = kwargs.get("suffix")
        return out if suffix is None else torch.cat((out, suffix), -1)[:, :x.shape[-1]]
    def grouped_norm(q, kv, q_weight, kv_weight, eps, *, kv_suffix=None):
        return norm(q, q_weight, eps), norm(kv, kv_weight, eps, suffix=kv_suffix)
    stub("rtp_llm.models_py.triton_kernels.common.layernorm_gated",
         fused_q_kv_rmsnorm=None, grouped_q_kv_rmsnorm=grouped_norm, independent_q_kv_rmsnorm=norm)
    from contextlib import nullcontext
    stub("rtp_llm.models_py.utils.module_profiler", profile_scope=lambda *a: nullcontext())
    stub("rtp_llm.ops", RoleType=SimpleNamespace(PREFILL=0), KvCacheDataType=None)
    routes = load_file("natural_routes", "models_py/mla_routes.py")
    modules["rtp_llm.models_py.mla_routes"] = routes
    with patch.dict(sys.modules, modules):
        return load_file("natural_dsa_module", "models_py/modules/kimi_k3/dsa.py")


def load_decoder():
    dependencies = {}
    def stub(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        dependencies[name] = module
    stub("rtp_llm.models.kimi_k3.kimi_k3_weight", KimiK3WeightNames=None)
    stub("rtp_llm.models_py.distributed.collective_torch", Group=SimpleNamespace(TP=0),
         _get_group=None, all_gather=None, reduce_scatter_bf16=None)
    stub("rtp_llm.models_py.model_desc.block_map", get_attention_inputs_value=lambda x: x.primary,
         get_group_tags_for_layers=None, get_layer_caches_for_groups=None,
         get_primary_attention_inputs=lambda x, cache: x.primary,
         select_attention_inputs_for_layer=lambda x, *a: x.primary,
         select_attention_inputs_for_tag=lambda x, *a: x,
         select_fmha_impl_for_group=lambda x, tag: x[tag], select_fmha_impl_for_layer=None)
    stub("rtp_llm.models_py.modules", KdaMetadata=SimpleNamespace, Embedding=None,
         MultimodalEmbeddingInjector=None, RMSNorm=None)
    stub("rtp_llm.models_py.model_desc.module_base", GptModelBase=torch.nn.Module)
    stub("rtp_llm.models_py.modules.factory.linear", LinearFactory=None)
    stub("rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear", CudaF16Linear=None)
    stub("rtp_llm.models_py.modules.kimi_k3.attention", KimiK3KDA=None, KimiK3MLA=None)
    stub("rtp_llm.models_py.modules.kimi_k3.moe", KimiK3LatentMoE=None)
    stub("rtp_llm.models_py.modules.kimi_k3.residual", KimiK3AttentionResidual=None)
    stub("rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill", prepare_paged_short_conv_metadata=None)
    stub("rtp_llm.models_py.triton_kernels.moe.activation", situ_and_mul=None)
    from contextlib import nullcontext
    stub("rtp_llm.models_py.utils.module_profiler", profile_scope=lambda *a: nullcontext())
    stub("rtp_llm.ops", HybridAttentionType=SimpleNamespace(LINEAR="linear"), RoleType=None)
    stub("rtp_llm.ops.compute_ops", PyModelInputs=object, PyModelOutputs=object)
    stub("rtp_llm.utils.model_weight", W=None)
    with patch.dict(sys.modules, dependencies):
        adapter = load_file("natural_public_adapter", "models_py/modules/kimi_k3/dsa_model_adapter.py")
        with patch.dict(sys.modules, {"rtp_llm.models_py.modules.kimi_k3.dsa_model_adapter": adapter}):
            return load_file("natural_public_decoder", "models_py/model_desc/kimi_k3.py").KimiK3Model


class NaturalIntegrationTest(unittest.TestCase):
    def test_route_budget_uses_recommended_granularity_and_releases_retained_handle(self):
        helper = load_file("natural_route_budget", "models_py/modules/kimi_k3/natural_dsa.py")
        source = torch.empty(3, 5, dtype=torch.bfloat16)
        exchange = SimpleNamespace(source=source, max_rows=3, hidden_size=5,
                                   chunk_capacity=2, signal_pad_bytes=128, size=4)
        for granularity in (65536, 2097152):
            calls = []
            def retain(handle, pointer):
                self.assertEqual(pointer.value, source.data_ptr())
                handle._obj.value = 123
                return 0
            def properties(output, handle):
                self.assertEqual(handle.value, 123)
                self.assertEqual(__import__("ctypes").sizeof(output._obj), 32)
                return 0
            def query(output, properties, policy):
                self.assertEqual(policy.value, 1)
                output._obj.value = granularity
                return 0
            def release(handle):
                calls.append(handle.value)
                return 0
            driver = SimpleNamespace(cuMemRetainAllocationHandle=retain,
                cuMemGetAllocationPropertiesFromHandle=properties,
                cuMemGetAllocationGranularity=query, cuMemRelease=release)
            with patch("ctypes.CDLL", return_value=driver):
                budget, actual_granularity = helper.route_allocation_budget(exchange)
            self.assertEqual(actual_granularity, granularity)
            self.assertEqual(budget, 3 * granularity + 512)
            self.assertEqual(calls, [123])
            driver.cuMemGetAllocationGranularity = lambda *args: 7
            with patch("ctypes.CDLL", return_value=driver), self.assertRaisesRegex(RuntimeError, "driver error 7"):
                helper.route_allocation_budget(exchange)
            self.assertEqual(calls, [123, 123])

    def test_full_pre_norm_publication_and_last_hidden_keep_original_token_identity(self):
        cls = load_decoder()
        for rank in range(4):
            model = cls.__new__(cls)
            torch.nn.Module.__init__(model)
            query = fixture(rank, holes=True)
            natural = load_hybrid("natural_outer_layout").NaturalOuterLayout(query)
            plan = load_hybrid("natural_zigzag_plan").NaturalZigzagPlan(query.lengths, 4, rank)
            tags = torch.stack((torch.arange(24), torch.arange(24) + 100), -1).bfloat16()
            global_valid = natural.rankmajor_positions >= 0
            masked = tags.masked_fill(~global_valid[:, None], 0)
            zigzag = masked[query.restore.argsort()]
            start, stop = rank * 6, (rank + 1) * 6
            # Represents already injected multimodal rows; entry must mask and
            # route the complete hidden features before the first anchor.
            embedded = tags[query.restore.argsort()][start:stop].clone()
            events = []
            expected_natural_final = masked * 2 + 4
            expected_public = expected_natural_final[query.restore.argsort()]
            test = self
            class Exchange:
                def exchange(self, value, actual_plan, direction):
                    test.assertIs(actual_plan, plan)
                    events.append(direction)
                    if direction == "to_natural":
                        torch.testing.assert_close(value, zigzag[start:stop], rtol=0, atol=0)
                        return masked[start:stop].clone()
                    torch.testing.assert_close(value, expected_natural_final[start:stop], rtol=0, atol=0)
                    return expected_public[start:stop].clone()
            class Layer:
                layer_type = "dsa"
                attention = SimpleNamespace(indexer=object())
                def __call__(self, hidden, anchors, fmha, cache, inputs, metadata, valid):
                    test.assertIs(metadata.query_layout, query)
                    test.assertIs(getattr(metadata, "outer_layout", None), natural)
                    test.assertIs(valid, natural.local_valid)
                    torch.testing.assert_close(hidden, masked[start:stop], rtol=0, atol=0)
                    anchors[:, 0].copy_(hidden)
                    events.append("layer")
                    return hidden + 1
            model._query_layout, model._outer_layout, model._routing_plan = query, natural, plan
            model._row_exchange, model.layers = Exchange(), [Layer()]
            model.tp_size, model.tp_rank, model.num_blocks = 4, rank, 1
            model.kv_cache = None
            model.final_residual = lambda hidden, anchors: hidden + anchors[:, 0] + 3
            model._mtp_prefill_hidden_buffer = torch.empty(8, 2, dtype=torch.bfloat16)
            model._mtp_prefill_hidden_valid_rows = 0
            inputs = SimpleNamespace(input_ids=torch.arange(6), primary=SimpleNamespace(
                valid_token_mask=None, is_prefill=True, is_target_verify=False))
            output = model._forward_layers(embedded, inputs, {"full": object(), "indexer_kv": object()})
            torch.testing.assert_close(output, expected_public[start:stop], rtol=0, atol=0)
            torch.testing.assert_close(model.get_mtp_target_hidden_states(-1), expected_public[start:stop], rtol=0, atol=0)
            self.assertEqual(events, ["to_natural", "layer", "to_zigzag"])
            query.prepare_last_hidden_rows()
            last = query.local_last_hidden(output)
            expected = expected_natural_final[torch.tensor([10, 20])].clone()
            expected[~query.last_is_mine] = 0
            torch.testing.assert_close(last, expected, rtol=0, atol=0)

    def test_budget_charges_routes_and_invalidates_plan_for_layout_capacity(self):
        native = ModuleType("rtp_llm.models_py.modules.dsv4.indexer_topk")
        native.native_exact_topk_enabled = lambda: False
        native.native_candidate_selection_supported = lambda *a, **kw: False
        native.native_page_rr_selection_supported = lambda *a, **kw: False
        ops = ModuleType("rtp_llm.ops")
        ops.KvCacheDataType = SimpleNamespace(FP8="fp8")
        package = ModuleType("natural_budget")
        directory = ROOT / "models_py/modules/factory/attention/cuda_mla_impl"
        package.__path__ = [str(directory.parent)]
        cuda_package = ModuleType("natural_budget.cuda_mla_impl")
        cuda_package.__path__ = [str(directory)]
        with patch.dict(sys.modules, {
            native.__name__: native,
            ops.__name__: ops,
            package.__name__: package,
            cuda_package.__name__: cuda_package,
        }):
            workspace = load_file("natural_budget.cuda_mla_impl.dsa_kv_workspace", directory / "dsa_kv_workspace.py")
            execution = load_file("natural_budget.cuda_mla_impl.dsa_kv_chunk_execution", directory / "dsa_kv_chunk_execution.py")
            owner = workspace.DsaWorkspaceOwner(device="cpu", hidden_dim=16, q_rank=8, tp_size=4)
            owner.prefill_query_cp = True
            before = owner.external_projection_bytes(12, indexer=True)
            owner.route_bytes = 123456
            after = owner.external_projection_bytes(12, indexer=True)
            self.assertEqual(after - before, owner.route_bytes, "persistent route allocation is absent from budget")
            owner.layout_bytes = 1234
            params = SimpleNamespace(workspace_owner=owner, is_prefill=True, is_cuda_graph=False,
                query_positions=torch.arange(12), positions_d=torch.arange(12),
                kv_lens_host=[111, 705], shard_size=4, shard_rank=0,
                page_size=8, kernel_page_size=4, block_table=torch.ones(2, 100))
            key = execution.DsaKVExecution.planning_key(params, "indexer")
            owner.layout_bytes += 64
            self.assertNotEqual(key, execution.DsaKVExecution.planning_key(params, "indexer"))
            key = execution.DsaKVExecution.planning_key(params, "indexer")
            owner.route_capacity += 32
            self.assertNotEqual(key, execution.DsaKVExecution.planning_key(params, "indexer"))

    def test_natural_producer_slots_keep_zigzag_queries_and_persistent_cache_bytes(self):
        package = "natural_cpu_page"
        page_dir = ROOT / "models_py/modules/factory/attention/cuda_mla_impl"
        mod = ModuleType(package)
        mod.__path__ = [str(page_dir)]
        collective = ModuleType("rtp_llm.models_py.distributed.collective_torch")
        collective.Group = SimpleNamespace(TP=0)
        collective.all_gather_into = None
        distributed = ModuleType("rtp_llm.models_py.distributed")
        distributed.collective_torch = collective
        page_meta = ModuleType(package + ".page_rr_mla_metadata")
        page_meta.PageRRMlaMetadata = None
        from contextlib import nullcontext
        profiler = ModuleType("rtp_llm.models_py.utils.module_profiler")
        profiler.profile_scope = lambda *a: nullcontext()
        deps = {package: mod, package + ".page_rr_mla_metadata": page_meta,
                collective.__name__: collective, distributed.__name__: distributed,
                profiler.__name__: profiler}
        event = SimpleNamespace(record=lambda *a: None)
        with patch.dict(sys.modules, deps), patch.object(torch.cuda, "Event", return_value=event), \
                patch.object(torch.cuda, "current_stream", return_value=None):
            cls = load_file(package + ".dsa_page_rr_params", page_dir / "dsa_page_rr_params.py").DsaPageRRParams
            for lengths in ([11, 5], [29, 45]):
                size, prefixes = 4, [100, 700]
                padded = [(n + 7) // 8 * 8 for n in lengths]
                rankmajor = []
                for rank in range(size):
                    start, rows = 0, []
                    for n in padded:
                        width = n // 8
                        rows += list(range(start + rank * width, start + (rank + 1) * width))
                        rows += list(range(start + n - (rank + 1) * width, start + n - rank * width))
                        start += n
                    rankmajor += rows
                restore = torch.tensor(rankmajor).argsort()
                padding = torch.cat([torch.arange(p) < n for p, n in zip(padded, lengths)])
                validity = torch.ones(sum(lengths), dtype=torch.bool)
                validity[[2, lengths[0] + 1]] = False
                info = SimpleNamespace(prefill_actual_input_lengths_cpu=torch.tensor(lengths),
                    prefill_prefix_lengths_cpu=torch.tensor(prefixes),
                    prefill_qkv_restore_indice=restore, prefill_qkv_padding_mask=padding)
                tags = torch.zeros(sum(padded), 2, dtype=torch.bfloat16)
                cursor = 0
                for request, (length, pad) in enumerate(zip(lengths, padded)):
                    tags[cursor:cursor + length, 0] = request + 1
                    tags[cursor:cursor + length, 1] = torch.arange(length)
                    cursor += pad
                for rank in range(size):
                    query = load_hybrid("query_layout").QueryLayout(info, size, rank, "cpu", validity)
                    producer = load_hybrid("natural_outer_layout").NaturalOuterLayout(query)
                    owner = SimpleNamespace(prefill_query_cp=True, query_layout=query, producer_layout=producer,
                        index_dim=2, latent_dim=1, suffix_dim=1, device="cpu",
                        budgets={"indexer": 1 << 30, "sfa": 1 << 30},
                        phase=lambda name: object(), external_projection_bytes=lambda *a, **kw: 0,
                        other_resident_bytes=lambda phase: 0)
                    parallel = SimpleNamespace(tp_size=size, tp_rank=rank,
                        prefill_cp_config=SimpleNamespace(kv_cache_sharded=True, is_enabled=lambda: True))
                    params = cls(SimpleNamespace(tokens_per_block=8, kernel_tokens_per_block=4), parallel, workspace_owner=owner)
                    table = torch.arange(1, 201, dtype=torch.int32).reshape(2, 100)
                    params.prepare(SimpleNamespace(is_prefill=True, is_target_verify=False,
                        is_mtp_draft_update=False, kv_cache_kernel_block_id_device=table))
                    self.assertIs(getattr(params, "producer_layout", None), producer,
                                  "producer metadata still aliases Zigzag queries")
                    self.assertIs(params.query_positions, query.local_positions)
                    self.assertIs(params.producer_positions, producer.local_positions)
                    cache = torch.full((804, 2), -99, dtype=torch.bfloat16)
                    expected = cache.clone()
                    cursor = 0
                    for request, (length, pad, prefix) in enumerate(zip(lengths, padded, prefixes)):
                        for position in range(length):
                            if position == (2 if request == 0 else 1):
                                continue
                            absolute = prefix + position
                            if absolute // 8 % size == rank:
                                local = absolute // (8 * size) * 8 + absolute % 8
                                slot = int(table[request, local // 4]) * 4 + local % 4
                                expected[slot] = torch.tensor([request + 1, position]).bfloat16()
                        cursor += pad
                    ingress = params.ingress_plans["sfa"]
                    for start, count, slots in ingress.blocks:
                        payload = torch.zeros(size * ingress.capacity, 2, dtype=torch.bfloat16)
                        for peer in range(size):
                            source = tags[peer * producer.local_rows + start:peer * producer.local_rows + start + count]
                            payload[peer * ingress.capacity:peer * ingress.capacity + count] = source
                        live = slots >= 0
                        cache[slots[live]] = payload[live]
                    torch.testing.assert_close(cache.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)

    def test_prepare_separates_outer_and_query_metadata_without_old_kda_scratch(self):
        dependencies = {}
        def stub(name, **values):
            mod = ModuleType(name)
            mod.__dict__.update(values)
            dependencies[name] = mod
        stub("rtp_llm.models_py.model_desc.block_map",
             get_primary_attention_inputs=lambda inputs, cache: inputs.primary,
             get_attention_inputs_value=lambda inputs: inputs.primary)
        stub("rtp_llm.ops", HybridAttentionType=SimpleNamespace(LINEAR="linear"))
        for name in ("query_layout", "natural_outer_layout", "natural_zigzag_plan"):
            dependencies["rtp_llm.models_py.modules.hybrid." + name] = load_hybrid(name)
        dependencies["rtp_llm.models_py.modules.kimi_k3.natural_dsa"] = load_file(
            "natural_helpers", "models_py/modules/kimi_k3/natural_dsa.py")
        captures = []
        stub("rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl",
             SparseMlaPageRRImpl=lambda *args, **kwargs: captures.append(kwargs["workspace_owner"]))
        with patch.dict(sys.modules, dependencies):
            cls = load_file("rtp_llm.models_py.modules.kimi_k3.natural_adapter_test", "models_py/modules/kimi_k3/dsa_model_adapter.py").KimiK31ModelAdapter
            model = cls()
            original = fixture(1, padded=False)
            info = SimpleNamespace(prefill_actual_input_lengths_cpu=original.lengths_host,
                prefill_prefix_lengths_cpu=torch.tensor(original.prefixes),
                prefill_qkv_restore_indice=original.restore, prefill_qkv_padding_mask=original.padding_mask)
            primary = SimpleNamespace(is_prefill=True, is_target_verify=False, is_mtp_draft_update=False,
                                      valid_token_mask=None, context_parallel_info=info)
            model.config = SimpleNamespace(is_mtp=False)
            model.prefill_query_cp, model.tp_size, model.tp_rank = True, 4, 1
            model._outer_natural_enabled = True
            model._query_layout_buffers, model._outer_layout_buffers = {}, {}
            model.kv_cache = model.weight = model.fmha_config = None
            model.parallelism_config = SimpleNamespace(prefill_cp_config=SimpleNamespace(preserve_tp_layout=True))
            model.layers = [SimpleNamespace(layer_type="linear", attention=SimpleNamespace(
                input=SimpleNamespace(weight=torch.empty(17, 4)), width=9, _fp8_output_norm=False))]
            plans = []
            model._row_exchange = SimpleNamespace(chunk_capacity=4, max_rows=16,
                prepare_plan=lambda plan, **kw: plans.append((plan, kw)))
            model.dsa_workspace = SimpleNamespace(route_chunk_rows=4)
            model._prepare_dsa(SimpleNamespace(primary=primary, input_ids=torch.arange(4)), False)
            self.assertIsNotNone(getattr(model, "_outer_layout", None), "natural metadata was not prepared")
            self.assertIs(model.dsa_workspace.query_layout, model._query_layout)
            self.assertIs(model.dsa_workspace.producer_layout, model._outer_layout)
            self.assertEqual(model._query_layout_buffers, {})
            self.assertEqual(model._outer_layout_buffers, {})
            self.assertEqual(len(plans), 1)

            self.assertEqual(model._outer_layout.local_positions.tolist(), [104, 105, 106, 107])
            self.assertEqual(model._query_layout.local_positions.tolist(), [102, 103, 112, 113])
            self.assertGreater(model.dsa_workspace.layout_bytes, model._query_layout.resident_bytes)
            primary.is_target_verify = True
            model._prepare_dsa(SimpleNamespace(primary=primary, input_ids=torch.arange(4)), False)
            self.assertIsNone(model._outer_layout)
            self.assertIsNone(model.dsa_workspace.routing_plan)
            self.assertEqual(len(plans), 1)
            primary.is_target_verify = False
            info.prefill_actual_input_lengths_cpu = torch.tensor([0, 16])
            with self.assertRaisesRegex(ValueError, "nonempty real requests"):
                model._prepare_dsa(SimpleNamespace(primary=primary, input_ids=torch.arange(4)), False)
            info.prefill_actual_input_lengths_cpu = original.lengths_host
            model._outer_natural_enabled = False
            model._prepare_dsa(SimpleNamespace(primary=primary, input_ids=torch.arange(4)), False)
            self.assertIsNone(model._outer_layout)
            self.assertIs(model.dsa_workspace.producer_layout, model._query_layout)
            self.assertTrue(model._query_layout_buffers)
            self.assertEqual(len(plans), 1)

            # Cache hits retain owned static metadata but must still validate
            # the current producer's contents, even when its tensor is reused.
            model._outer_natural_enabled = True
            cached_query = next(iter(model._natural_layout_cache.values()))[0]
            model._prepare_dsa(SimpleNamespace(primary=primary, input_ids=torch.arange(4)), False)
            self.assertIs(model._query_layout, cached_query)
            saved = info.prefill_qkv_restore_indice.clone()
            info.prefill_qkv_restore_indice.copy_(saved.roll(1))
            with self.assertRaisesRegex(ValueError, "permutation"):
                model._prepare_dsa(SimpleNamespace(primary=primary, input_ids=torch.arange(4)), False)
            torch.testing.assert_close(cached_query.restore, saved, rtol=0, atol=0)
            info.prefill_qkv_restore_indice.copy_(saved)

    def test_query_chunks_fill_destination_slices_and_history_runs_once(self):
        from rtp_llm.models_py.modules.kimi_k3.natural_dsa import Fp8NaturalProjection, NaturalQGateProjection

        module = load_dsa()
        cls = module.KimiK31DSA
        self.assertTrue(hasattr(cls, "_forward_natural"), "natural DSA execution missing")
        torch.manual_seed(31)
        for fp8, chunk_rows in [(fp8, n) for fp8 in (False, True) for n in (1, 3, 5, 6, 9)]:
            model = cls.__new__(cls)
            torch.nn.Module.__init__(model)
            model.q_rank, model.kv_rank, model.suffix_dim = 4, 2, 2
            model.heads, model.q_dim, model.v_dim, model.layer_idx = 2, 2, 2, 0
            model._fp8_output_gate = False
            def linear(i, o):
                return torch.nn.Linear(i, o, bias=False).bfloat16()
            model.natural_q_input = linear(4, 4)
            model.natural_kv_input = linear(4, 4)
            model.natural_gate_input = linear(4, 4)
            quantizations, encodings = [], {"q": [], "gate": []}
            if fp8:
                # Real routed control flow; CPU arithmetic replaces only the
                # unavailable CUDA projection, exposing encoding reuse/ordering.
                class CpuFp8Projection(Fp8NaturalProjection):
                    def __init__(self, source, name):
                        torch.nn.Module.__init__(self)
                        self.weight, self.name = source.weight, name
                    def quantize_input(self, value):
                        quantizations.append((self.name, value.shape[0]))
                        return value.to(torch.float8_e4m3fn), torch.ones(value.shape[0], 1)
                    def forward_quantized(self, values, scales):
                        encodings[self.name].append((values, scales))
                        output = (values.float() @ self.weight.float().T).bfloat16()
                        if self.name == "gate":
                            produced["gate"] = output
                        return output
                model.natural_q_input = CpuFp8Projection(model.natural_q_input, "q")
                model.natural_gate_input = CpuFp8Projection(model.natural_gate_input, "gate")
                class CpuQGate(torch.nn.Module):
                    def forward(inner, value):
                        values, scales = model.natural_q_input.quantize_input(value)
                        encodings["q"].append((values, scales))
                        encodings["gate"].append((values, scales))
                        weight = torch.cat((model.natural_q_input.weight,
                                            model.natural_gate_input.weight), 0)
                        q, gate = (values.float() @ weight.float().T).bfloat16().split((4, 4), -1)
                        produced["gate"] = gate
                        return q, gate
                model.natural_q_gate_input = CpuQGate()
            else:
                model.natural_q_gate_input = NaturalQGateProjection(
                    model.natural_q_input, model.natural_gate_input)
            model.q_b, model.output = linear(4, 4), linear(4, 4)
            model.q_norm = SimpleNamespace(weight=torch.ones(4).bfloat16(), variance_epsilon=1e-5)
            model.kv_norm = SimpleNamespace(weight=torch.ones(2).bfloat16(), variance_epsilon=1e-5)
            hidden = torch.arange(24).reshape(6, 4).bfloat16() / 32
            permutation = torch.tensor([4, 1, 5, 0, 3, 2])
            query_hidden = hidden[permutation]
            events, received, produced = [], {}, {}
            model.q_b.register_forward_hook(lambda m, a, value: produced.update(query=value))
            model.natural_gate_input.register_forward_hook(lambda m, a, value: produced.update(gate=value))
            if not fp8:
                model.natural_q_gate_input.register_forward_hook(
                    lambda m, a, value: produced.update(gate=value[1]))
            gate_sigmoid_mul = module.gate_sigmoid_mul

            def checked_gate(output, gate, mask):
                if chunk_rows >= 6:
                    self.assertEqual(gate.untyped_storage().data_ptr(), produced["gate"].untyped_storage().data_ptr())
                return gate_sigmoid_mul(output, gate, mask)

            class Exchange:
                def consume_chunks(self, values, plan, direction, *, chunk_rows, consumer,
                                   before_consume=None):
                    self_outer.assertEqual(direction, "to_zigzag")
                    if before_consume is not None:
                        before_consume()
                    for start in range(0, 6, chunk_rows):
                        slot = values[permutation[start:start + chunk_rows]]
                        consumer(start, slot)
                        slot.fill_(float("nan"))
                def exchange(self, values, plan, direction):
                    self_outer.assertEqual(direction, "to_natural")
                    return values[permutation.argsort()].clone()
            self_outer = self
            def key_write(key, cache, params):
                events.append("key")
                received["key"] = key.clone()
            def topk(query, weights, *args):
                if chunk_rows >= 6:
                    self.assertEqual(query.untyped_storage().data_ptr(), produced["index_query"].untyped_storage().data_ptr())
                    self.assertEqual(weights.untyped_storage().data_ptr(), produced["weights"].untyped_storage().data_ptr())
                events.extend(["index_history"] * 3)
                received["index_query"], received["weights"] = query.clone(), weights.clone()
                return torch.zeros(6, 2, dtype=torch.int32)
            def encode(query):
                encoded = query.to(torch.float8_e4m3fn)
                produced["index_query"] = encoded
                return encoded, torch.ones(query.shape[0], 2, 1)
            def head_weights(hidden, scale, **kwargs):
                value = hidden[:, :2].float().unsqueeze(-1) * scale
                produced["weights"] = value
                return value
            model.indexer = SimpleNamespace(
                index_n_heads=2, index_head_dim=2,
                project_key=lambda x: x[:, :2].clone(),
                project_query=lambda q, **kw: q.reshape(-1, 2, 2),
                project_head_weights=head_weights,
                indexer_op=SimpleNamespace(write_key=key_write,
                    encode_query=encode),
                _compute_topk=topk,
            )
            def attend(q, latent, suffix, cache, layer, indices):
                if chunk_rows >= 6:
                    self.assertEqual(q.untyped_storage().data_ptr(), produced["query"].untyped_storage().data_ptr())
                received["query"], received["latent"], received["suffix"] = q.clone(), latent.clone(), suffix.clone()
                events.extend(["sfa_history"] * 4 + ["v_up", "sfa_publish"])
                return q.clone()
            owner = SimpleNamespace(row_exchange=Exchange(), routing_plan=object(), route_chunk_rows=chunk_rows)
            sfa = SimpleNamespace(forward=attend, workspace_owner=owner)
            inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=object(),
                cache_store_writer=SimpleNamespace(write=lambda *args: events.append("index_publish")))
            index_fmha = SimpleNamespace(fmha_params=SimpleNamespace(is_prefill=True),
                                        cp_params=object(), attn_inputs=inputs, is_sparse=lambda: True)
            layout = SimpleNamespace(local_rows=6, all_valid=False,
                local_valid=torch.tensor([True, False, True, True, False, True]))
            with patch.object(module, "gate_sigmoid_mul", checked_gate):
                result = model._forward_natural(hidden, {"full": sfa, "indexer_kv": index_fmha},
                                               {"full": object(), "indexer_kv": object()},
                                               SimpleNamespace(query_layout=layout))
            if fp8:
                chunks = (6 + chunk_rows - 1) // chunk_rows
                self.assertEqual(len(quantizations), chunks)
                self.assertTrue(all(name == "q" for name, rows in quantizations))
                self.assertEqual(len(encodings["q"]), chunks)
                self.assertEqual(len(encodings["gate"]), chunks)
                for q_encoding, gate_encoding in zip(encodings["q"], encodings["gate"]):
                    self.assertIs(q_encoding[0], gate_encoding[0])
                    self.assertIs(q_encoding[1], gate_encoding[1])
            torch.testing.assert_close(received["key"], hidden[:, :2], rtol=0, atol=0)
            torch.testing.assert_close(received["weights"], query_hidden[:, :2].float().unsqueeze(-1), rtol=0, atol=0)
            q = model.natural_q_input(query_hidden)
            q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-5)).bfloat16()
            torch.testing.assert_close(
                received["index_query"].view(torch.uint8),
                q.reshape(6, 2, 2).to(torch.float8_e4m3fn).view(torch.uint8),
                rtol=0, atol=0,
            )
            expected_query = model.q_b(q).reshape(6, 2, 2)
            torch.testing.assert_close(received["query"], expected_query, rtol=0, atol=0)
            gate = model.natural_gate_input(query_hidden)
            expected = (expected_query.reshape(6, 4) * gate.sigmoid()).masked_fill(~layout.local_valid[:, None], 0)
            expected = model.output(expected)[permutation.argsort()]
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertEqual(events.count("key"), 1)
            self.assertEqual(events.count("index_history"), 3)
            self.assertEqual(events.count("sfa_history"), 4)
            for event in ("v_up", "sfa_publish", "index_publish"):
                self.assertEqual(events.count(event), 1)
            def unexpected_query(*args, **kwargs):
                self.fail("absent-cache/warmup path must not project Indexer queries")
            model.indexer.project_query = unexpected_query
            for cache_case in (None, {"full": object(), "indexer_kv": object()}):
                if cache_case is not None:
                    index_fmha.is_sparse = lambda: False
                with patch.object(module, "gate_sigmoid_mul", checked_gate):
                    again = model._forward_natural(hidden, {"full": sfa, "indexer_kv": index_fmha},
                                                   cache_case, SimpleNamespace(query_layout=layout))
                torch.testing.assert_close(again, expected, rtol=0, atol=0)
                # The previous return is durable through another serial use
                # of the same transport and destination construction.
                torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_restore_validation_rejects_equal_shape_wrong_permutation(self):
        helper_path = ROOT / "models_py/modules/kimi_k3/natural_dsa.py"
        self.assertTrue(helper_path.exists(), "natural integration contract missing")
        helper = load_file("natural_contract", helper_path)
        Plan = load_hybrid("natural_zigzag_plan").NaturalZigzagPlan
        for rank in range(4):
            layout = fixture(rank, holes=True)
            plan = Plan(layout.lengths, 4, rank)
            helper.validate_shared_restore(layout, plan)
            layout.restore[[0, 1]] = layout.restore[[1, 0]]
            with self.assertRaisesRegex(ValueError, "restore"):
                helper.validate_shared_restore(layout, plan)

    def test_natural_outer_is_default_for_query_cp_target(self):
        block_map = ModuleType("rtp_llm.models_py.model_desc.block_map")
        block_map.get_attention_inputs_value = None
        block_map.get_primary_attention_inputs = None
        ops = ModuleType("rtp_llm.ops")
        ops.HybridAttentionType = SimpleNamespace(LINEAR="linear")
        with patch.dict(sys.modules, {
            block_map.__name__: block_map,
            ops.__name__: ops,
        }):
            adapter = load_file(
                "natural_default_adapter",
                "models_py/modules/kimi_k3/dsa_model_adapter.py",
            ).KimiK31ModelAdapter
            for query_cp in (False, True):
                for is_mtp in (False, True):
                    with self.subTest(query_cp=query_cp, is_mtp=is_mtp):
                        model = adapter()
                        model.prefill_query_cp = query_cp
                        model.config = SimpleNamespace(
                            is_mtp=is_mtp,
                            k3_runtime_config=SimpleNamespace(dsa=None),
                        )
                        model.register_buffer = lambda *args, **kwargs: None
                        model._setup_dsa()
                        self.assertEqual(
                            model._outer_natural_enabled, query_cp and not is_mtp
                        )

    def test_prepared_indexer_runs_once_and_publishes_after_success(self):
        routes = load_file("natural_routes", "models_py/mla_routes.py")
        self.assertTrue(hasattr(routes, "run_prepared_sparse_indexer"),
                        "prepared Indexer publication boundary missing")
        events = []
        query = torch.arange(24).reshape(3, 2, 4).bfloat16()
        weights = torch.ones(3, 2, 1)
        cache = torch.zeros(3, 4)
        expected = torch.tensor([[100], [101], [700]], dtype=torch.int32)

        def compute(q, w, c, params, inputs, cp):
            self.assertIs(q, query)
            self.assertIs(w, weights)
            self.assertIs(c, cache)
            events.append("compute")
            return expected

        inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=object(),
                                 cache_store_writer=SimpleNamespace(write=lambda *args: events.append("publish")))
        fmha = SimpleNamespace(attn_inputs=inputs, fmha_params=object(), cp_params=object(),
                               is_sparse=lambda: True)
        indexer = SimpleNamespace(_compute_topk=compute)
        result = routes.run_prepared_sparse_indexer(indexer, query, weights, cache, fmha)
        self.assertIs(result, expected)
        self.assertEqual(events, ["compute", "publish"])
        self.assertIsNone(routes.run_prepared_sparse_indexer(indexer, query, weights, None, fmha))
        self.assertEqual(events, ["compute", "publish"])
        fmha.is_sparse = lambda: False
        self.assertIsNone(routes.run_prepared_sparse_indexer(indexer, query, weights, cache, fmha))
        self.assertEqual(events, ["compute", "publish", "publish"])


if __name__ == "__main__":
    unittest.main()
