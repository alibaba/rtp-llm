import ast
import json
from pathlib import Path


RTP_LLM_DIR = Path(__file__).resolve().parents[1]
ATTENTION_DIR = RTP_LLM_DIR / "models_py/modules/factory/attention"
MODEL_WEIGHT_INFO = RTP_LLM_DIR / "model_loader/model_weight_info.py"
HW_KERNEL_CONFIG = RTP_LLM_DIR / "cpp/config/ConfigModules.h"
CONFIG_BINDINGS = RTP_LLM_DIR / "cpp/pybind/ConfigInit.cc"
FP8_PTPC_LINEAR = (
    RTP_LLM_DIR
    / "models_py/modules/factory/linear/impl/rocm/fp8_ptpc_linear.py"
)
CUDA_LINEAR_REGISTRY = (
    RTP_LLM_DIR / "models_py/modules/factory/linear/impl/cuda/__init__.py"
)
DEVICE_IMPL = RTP_LLM_DIR / "device/device_impl.py"
MIXED_FP4_WEIGHT = RTP_LLM_DIR / "model_loader/mixed_fp4_quant_weight.py"


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_cuda_f16_linear_registers_before_optional_quantized_backends():
    tree = ast.parse(CUDA_LINEAR_REGISTRY.read_text())
    cuda_branch = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "is_cuda()"
    )

    statements = [ast.unparse(node) for node in cuda_branch.body]
    f16_register = statements.index("LinearFactory.register(CudaF16Linear)")
    fp8_import = next(
        index
        for index, statement in enumerate(statements)
        if "fp8_gemm_linear" in statement
    )

    assert f16_register < fp8_import


def test_tbstars_fp8_ptpc_keeps_raw_weights_and_selects_reference_linear():
    tree = ast.parse(MODEL_WEIGHT_INFO.read_text())
    deploy_info = _find_class(tree, "ModelDeployWeightInfo")
    configure_source = ast.unparse(
        _find_method(deploy_info, "_configure_legacy_fp8_ptpc")
    )
    init_source = ast.unparse(_find_method(deploy_info, "__init__"))

    assert "model_config.quant_algo.isFp8PTPC()" in configure_source
    assert "model_config.model_type == 'tbstars'" in configure_source
    assert "model_config.hidden_size == 1024" in configure_source
    assert "model_config.inter_size == 2816" in configure_source
    assert "hw_kernel_config.force_legacy_fp8_ptpc" in configure_source
    assert "hw_kernel_config.use_swizzleA = False" not in configure_source
    assert (
        "self._configure_legacy_fp8_ptpc(model_config, hw_kernel_config)"
        in init_source
    )
    assert (
        "self._use_swizzleA = hw_kernel_config.use_swizzleA"
        in init_source
    )

    if HW_KERNEL_CONFIG.exists():
        assert "bool        force_legacy_fp8_ptpc" in HW_KERNEL_CONFIG.read_text()
    if CONFIG_BINDINGS.exists():
        assert (
            '.def_readwrite("force_legacy_fp8_ptpc"'
            in CONFIG_BINDINGS.read_text()
        )

    linear_tree = ast.parse(FP8_PTPC_LINEAR.read_text())
    reference_support = ast.unparse(
        _find_method(_find_class(linear_tree, "RocmFp8PTPCLinearReference"), "can_handle")
    )
    reference_init = ast.unparse(
        _find_method(_find_class(linear_tree, "RocmFp8PTPCLinearReference"), "__init__")
    )
    no_swizzle_support = ast.unparse(
        _find_method(_find_class(linear_tree, "RocmFp8PTPCLinearNoSwizzle"), "can_handle")
    )
    hipblas_support = ast.unparse(
        _find_method(_find_class(linear_tree, "RocmFp8PTPCLinearWithSwizzle"), "can_handle")
    )
    assert "hw_kernel_config.force_legacy_fp8_ptpc" in reference_support
    assert (
        "weight.reshape(self.output_size, self.hidden_size)" in reference_init
    )
    assert "self._shuffle_weight_for_cktile(checkpoint_weight)" in reference_init
    assert "self._as_hipb_scale_b(weight_scales, self.output_size)" in reference_init
    reference_forward = ast.unparse(
        _find_method(
            _find_class(linear_tree, "RocmFp8PTPCLinearReference"), "forward"
        )
    )
    assert "self._quantize_input(input)" in reference_forward
    assert "gemm_a8w8_bpreshuffle_cktile" in reference_forward
    assert "not hw_kernel_config.force_legacy_fp8_ptpc" in no_swizzle_support
    assert "not hw_kernel_config.force_legacy_fp8_ptpc" in hipblas_support

    device_tree = ast.parse(DEVICE_IMPL.read_text())
    force_branches = [
        node
        for node in ast.walk(device_tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not hw_kernel_config.force_legacy_fp8_ptpc"
    ]
    assert len(force_branches) == 1
    branch_source = ast.unparse(force_branches[0])
    assert "swizzle_tensor" in branch_source
    assert "shuffle_gemm_weight" in branch_source


def test_fused_rope_calls_support_old_arm_and_new_x86_rtp_kernel_signatures():
    tree = ast.parse((RTP_LLM_DIR / "ops/fused_rope_kvcache_op.py").read_text())
    prefill_arg_source = ast.unparse(_find_function(tree, "_prefill_position_ids_arg"))
    assert "_get_fused_rope_kvcache().prefill_fused_rope_kvcache" in prefill_arg_source
    assert "'position_ids' in inspect.signature(fn).parameters" in prefill_arg_source
    assert "else 'cp_position_ids'" in prefill_arg_source

    decode_abi_source = ast.unparse(_find_function(tree, "_decode_has_cu_seqlens"))
    assert "_get_fused_rope_kvcache().decode_fused_rope_kvcache" in decode_abi_source
    assert "'cu_seqlens' in inspect.signature(fn).parameters" in decode_abi_source

    prefill_prepare = _find_method(
        _find_class(tree, "FusedRopeKVCachePrefillOpBase"), "prepare"
    )
    prepare_source = ast.unparse(prefill_prepare)
    assert "_prefill_position_ids_arg() == 'position_ids'" in prepare_source
    assert "attn_inputs.context_parallel_info.prefill_shuffle_indices" in prepare_source

    prefill_forward = _find_method(
        _find_class(tree, "FusedRopeKVCachePrefillOpBase"), "_forward"
    )
    prefill_call = next(
        node
        for node in ast.walk(prefill_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prefill_fused_rope_kvcache"
    )
    dynamic_keywords = [
        keyword for keyword in prefill_call.keywords if keyword.arg is None
    ]
    assert len(dynamic_keywords) == 1
    assert ast.unparse(dynamic_keywords[0].value) == (
        "{_prefill_position_ids_arg(): params.position_ids}"
    )

    decode_forward = _find_method(
        _find_class(tree, "FusedRopeKVCacheDecodeOp"), "forward"
    )
    decode_source = ast.unparse(decode_forward)
    assert "if _decode_has_cu_seqlens()" in decode_source
    assert "decode_args.extend([params.position_ids, params.sequence_lengths])" in decode_source
    assert "decode_args.append(params.sequence_lengths)" in decode_source
    call = next(
        node
        for node in ast.walk(decode_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decode_fused_rope_kvcache"
    )
    assert [ast.unparse(arg) for arg in call.args] == ["*decode_args"]
    assert any(keyword.arg == "tokens_per_block" for keyword in call.keywords)


def test_paged_prefill_backends_require_kv_cache():
    native_tree = ast.parse(
        (ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text()
    )
    native_support = _find_method(
        _find_class(native_tree, "PyFlashinferPagedPrefillImpl"), "support"
    )
    native_source = ast.unparse(native_support)
    assert "attn_inputs.kv_cache_kernel_block_id is not None" in native_source
    assert "attn_inputs.kv_cache_kernel_block_id.numel() > 0" in native_source

    trt_tree = ast.parse((ATTENTION_DIR / "cuda_impl/trt.py").read_text())
    trt_support = _find_method(
        _find_class(trt_tree, "TRTLLMFMHAv2PagedPrefillOp"), "support"
    )
    trt_source = ast.unparse(trt_support)
    assert "attn_inputs.kv_cache_kernel_block_id is not None" in trt_source
    assert "attn_inputs.kv_cache_kernel_block_id.numel() > 0" in trt_source


def test_ragged_prefill_uses_flashinfer_cuda_graph_buffers():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    ragged_op = _find_class(tree, "PyFlashinferPrefillAttnOp")
    init_source = ast.unparse(_find_method(ragged_op, "__init__"))
    prepare_source = ast.unparse(_find_method(ragged_op, "prepare"))

    assert "use_cuda_graph=self.enable_cuda_graph" in init_source
    assert "qo_indptr_buf=qo_indptr" in init_source
    assert "kv_indptr_buf=kv_indptr" in init_source
    assert "causal=self.is_causal" in prepare_source
    assert "forbid_realloc" in prepare_source

    ragged_impl = _find_class(tree, "PyFlashinferPrefillImpl")
    support_cuda_graph = _find_method(ragged_impl, "support_cuda_graph")
    assert isinstance(support_cuda_graph.body[0], ast.Return)
    assert support_cuda_graph.body[0].value.value is True


def test_paged_cuda_graph_plan_matches_padded_query_buffer():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    paged_op = _find_class(tree, "PyFlashinferPrefillPagedAttnOp")
    init_source = ast.unparse(_find_method(paged_op, "__init__"))
    prepare_source = ast.unparse(
        _find_method(paged_op, "prepare")
    )

    assert "self.is_causal = attn_configs.is_causal" in init_source
    plan_source = ast.unparse(_find_method(paged_op, "_plan_prefill_wrapper"))
    assert "self._plan_prefill_wrapper(qo_indptr)" in prepare_source
    assert "causal=self.is_causal" in plan_source
    assert "qo_indptr = self.qo_indptr" in prepare_source
    assert "offsets + attn_inputs.input_lengths" not in prepare_source


def test_native_prefill_keeps_rope_without_writing_dummy_cache():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    base_impl = _find_class(tree, "PyFlashinferPrefillImplBase")
    forward_source = ast.unparse(_find_method(base_impl, "forward"))

    assert "if self.need_rope_kv_cache" in forward_source
    assert "if self.need_rope_kv_cache and kv_cache is not None" in forward_source
    assert "self.kv_cache_write_op.forward(key, value, kv_cache)" in forward_source


def test_embedding_cuda_graph_does_not_fabricate_kv_cache():
    runner_source = (RTP_LLM_DIR / "cpp/cuda_graph/cuda_graph_runner.cc").read_text()
    compact_source = "".join(runner_source.split())

    assert "if (!kv_cache_group_tags_.empty())" in runner_source
    assert "kv_cache_kernel_block_id_device=torch::empty({0}" in compact_source
    assert "kv_cache_kernel_block_id=torch::empty({0}" in compact_source


def test_decode_cuda_graph_passes_cache_group_tags_to_capture():
    source = (
        RTP_LLM_DIR / "cpp/cuda_graph/tests/cuda_graph_decode_padding.py"
    ).read_text()

    assert "self.kv_cache.group_tags" in source


def test_qwen35_sm100_cuda_graph_cases_match_validated_loader_environment():
    tree = ast.parse(
        (RTP_LLM_DIR / "test/smoke/suites/test_smoke_sm100_moe.py").read_text()
    )
    cases = ast.literal_eval(
        next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SMOKE_CASES"
                for target in node.targets
            )
        )
    )

    for name in (
        "next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
        "next_moe_nvfp4_cudagraph_tp2_sm100",
    ):
        assert "--load_method scratch" not in cases[name]["smoke_args"]
        assert "envs" not in cases[name]

    data_dir = RTP_LLM_DIR / "test/smoke/data/model/qwen35"
    dp2 = json.loads(
        (data_dir / "q_r_35b_nvfp4_py_dp2_ll_cg_sm100_arm.json").read_text()
    )["query_result"]
    tp2 = json.loads(
        (data_dir / "q_r_35b_nvfp4_py_tp2_cg_sm100_arm.json").read_text()
    )["query_result"]

    empty_think = "\n\n<think>\n\n</think>\n\nHello! I'm **Qwen3.5**, the latest large language"
    reasoning = "\n\n<think>\nOkay, the user asked me to introduce myself. Let me start by recalling my identity"
    assert dp2[0]["result"]["response"] == empty_think
    assert {item["response"] for item in dp2[1]["result"]["response_batch"]} == {
        reasoning
    }
    assert tp2[0]["result"]["response"] == empty_think
    assert {item["response"] for item in tp2[1]["result"]["response_batch"]} == {
        empty_think
    }


def test_sm100_head_dim_256_cuda_graph_keeps_validated_decode_backends():
    tree = ast.parse(
        (ATTENTION_DIR / "cuda_impl/trtllm_gen.py").read_text()
    )
    support_source = ast.unparse(
        _find_method(_find_class(tree, "FlashInferTRTLLMDecodeOp"), "support")
    )

    assert "attention_inputs.is_cuda_graph" not in support_source
    assert "self.head_dim == 256" not in support_source

    xqa_tree = ast.parse((ATTENTION_DIR / "cuda_impl/xqa.py").read_text())
    assert all(
        not (
            isinstance(node, ast.FunctionDef)
            and node.name == "_reject_sm100_head_dim_256_cuda_graph"
        )
        for node in xqa_tree.body
    )
    for class_name in ("XQAImpl", "XQADecodeImpl"):
        support_source = ast.unparse(
            _find_method(_find_class(xqa_tree, class_name), "support")
        )
        assert "_reject_sm100_head_dim_256_cuda_graph" not in support_source

    flashinfer_tree = ast.parse(
        (ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text()
    )
    decode_init = ast.unparse(
        _find_method(
            _find_class(flashinfer_tree, "PyFlashinferDecodeAttnOp"), "__init__"
        )
    )
    assert "determine_use_tensor_core_from_configs(attn_configs)" in decode_init
    assert "_use_tensor_core_decode" not in decode_init


def test_online_modelopt_fp4_moe_weights_keep_validated_checkpoint_layout():
    tree = ast.parse(MIXED_FP4_WEIGHT.read_text())
    mixed_fp4 = _find_class(tree, "MixedFp4Weight")

    assert all(
        not (
            isinstance(node, ast.FunctionDef)
            and node.name == "_get_moe_quant_ckpt_infos"
        )
        for node in mixed_fp4.body
    )

    w2_source = ast.unparse(_find_method(mixed_fp4, "_get_moe_w2_quant_weight"))
    assert "src_weight_info.weights[0].name[:-len(W_SUFFIX)]" in w2_source
    assert "stack_" in w2_source

    w1_source = ast.unparse(_find_method(mixed_fp4, "_get_moe_w1_quant_weight"))
    assert "for w in src_weight_info.weights" in w1_source
    assert "stack_moe_w1" in w1_source
    assert "stack_moe_w1_s2" in w1_source


def test_sm100_fp8_case_uses_tracked_main_golden_name():
    suite_tree = ast.parse(
        (RTP_LLM_DIR / "test/smoke/suites/test_smoke_sm100_dense.py").read_text()
    )
    cases = ast.literal_eval(
        next(
            node.value
            for node in suite_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SMOKE_CASES"
                for target in node.targets
            )
        )
    )

    task_info = cases["fp8_attention_sm100"]["task_info"]
    assert task_info.endswith("q_r_block_fp8_sm100_arm.json")
    task_info_path = RTP_LLM_DIR / "test/smoke" / task_info
    assert task_info_path.is_file()

    task_config = json.loads(task_info_path.read_text())
    for request in task_config["query_result"]:
        query = request["query"]
        assert "temperature" not in query
        assert query["extra_configs"]["top_k"] == 1


def test_qwen3_standalone_goldens_match_python_native_h20_outputs():
    source = (
        RTP_LLM_DIR / "models_py/standalone/test/qwen3_test.py"
    ).read_text()

    assert (
        "\u4f60\u597d\uff01\u6211\u662f\u4f60\u7684\u865a\u62df\u52a9\u624b\uff0c\u4e00\u4e2a\u4e13\u6ce8\u4e8e\u5e2e\u52a9\u4f60\u89e3\u51b3\u95ee\u9898\u548c\u63d0\u4f9b\u652f\u6301\u7684AI\u52a9\u624b"
        in source
    )
    assert (
        "\u4f60\u597d\uff01\u6211\u662f\u5c0f\u660e\u54e5\uff0c\u5f88\u9ad8\u5174\u89c1\u5230\u4f60\uff01"
        in source
    )
    assert (
        "3.9 \u548c 3.11 \u4e2d\uff0c**3.9 \u5927\u4e8e 3.11**\u3002"
        in source
    )
    assert "3.9\u6bd43.11111111\u5927" in source


def test_kimi_linear_openai_golden_accepts_only_observed_h20_choices():
    comparer_source = (RTP_LLM_DIR / "test/smoke/openai_comparer.py").read_text()
    assert '"choices_alternatives"' in comparer_source
    assert "alternative_choices == actual_choices" in comparer_source

    task_info = json.loads(
        (
            RTP_LLM_DIR
            / "test/smoke/data/model/kimi_linear/q_r_bf16_tp2_kernel_block_size_64.json"
        ).read_text()
    )
    result = task_info["query_result"][0]["result"]
    primary = result["choices"][0]["message"]["content"]
    alternatives = {
        choices[0]["message"]["content"]
        for choices in result["choices_alternatives"]
    }

    assert primary.endswith("\u8fd9\u4e9b\u65b9\u6cd5")
    assert alternatives == {
        "\u5f53\u7136\u53ef\u4ee5\uff01\u4ee5\u4e0b\u662f\u56f4\u7ed5\u201c\u82f1\u8bed\u542c\u529b\u65b9\u6cd5\u201d\u4e3b\u9898\u63d0\u51fa\u7684\u51e0\u4e2a\u6709\u6df1\u5ea6\u3001\u542f\u53d1\u6027\u7684\u95ee\u9898\uff0c\u9002\u5408\u7528\u4e8e\u8bfe\u5802\u8ba8\u8bba\u3001\u81ea\u6211\u53cd\u601d\u6216\u6559\u5b66\u7814\u7a76\uff1a\n\n---\n\n### \u4e00\u3001\u65b9\u6cd5\u7c7b\u95ee\u9898\n1. **\u4f60\u901a\u5e38\u5982\u4f55\u8bad\u7ec3\u82f1\u8bed\u542c\u529b\uff1f\u4f60\u89c9\u5f97\u81ea\u5df1\u6700\u6709\u6548\u7684\u65b9\u6cd5\u662f\u4ec0\u4e48\uff1f\u4e3a\u4ec0\u4e48\uff1f**\n2. **\u4f60\u662f\u5426\u5c1d\u8bd5\u8fc7\u201c\u7cbe\u542c\u201d\u548c\u201c\u6cdb\u542c\u201d\u7ed3\u5408\u7684\u65b9\u6cd5\uff1f\u4f60\u89c9\u5f97\u54ea\u79cd\u5bf9\u4f60\u5e2e\u52a9\u66f4\u5927\uff1f**\n3. **\u4f60\u662f\u5426\u4f7f\u7528\u8fc7\u201c\u542c\u5199\u201d\u6216\u201c\u8ddf\u8bfb\u201d\u7ec3\u4e60\uff1f\u8fd9\u4e9b\u65b9\u6cd5\u5bf9\u4f60\u63d0\u5347",
        "\u5f53\u7136\u53ef\u4ee5\uff01\u4ee5\u4e0b\u662f\u56f4\u7ed5\u201c\u82f1\u8bed\u542c\u529b\u65b9\u6cd5\u201d\u4e3b\u9898\u63d0\u51fa\u7684\u51e0\u4e2a\u6709\u6df1\u5ea6\u3001\u542f\u53d1\u6027\u7684\u95ee\u9898\uff0c\u9002\u5408\u7528\u4e8e\u8bfe\u5802\u8ba8\u8bba\u3001\u81ea\u6211\u53cd\u601d\u6216\u6559\u5b66\u7814\u7a76\uff1a\n\n---\n\n### **1. \u8f93\u5165\u4e0e\u7406\u89e3\u7684\u5173\u7cfb**\n> \u201c\u5728\u82f1\u8bed\u542c\u529b\u4e2d\uff0c\u2018\u542c\u61c2\u2019\u662f\u5426\u4e00\u5b9a\u610f\u5473\u7740\u2018\u7406\u89e3\u2019\uff1f\u8bf7\u7ed3\u5408\u4f60\u7684\u5b66\u4e60\u7ecf\u9a8c\uff0c\u8c08\u8c08\u4f60\u5bf9\u2018\u53ef\u7406\u89e3\u8f93\u5165\uff08comprehensible input\uff09\u2019\u8fd9\u4e00\u6982\u5ff5\u7684\u7406\u89e3\u3002\u201d\n\n---\n\n### **2. \u7cbe\u542c\u4e0e\u6cdb\u542c\u7684\u5e73\u8861**\n> \u201c\u4f60\u8ba4\u4e3a\u7cbe\u542c",
    }

    batch = task_info["query_result"][1]["result"]["response_batch"]
    assert len(batch) == 3
    assert {
        response["response"] for response in batch
    } == {"\n\nCAP theorem states that any distributed system can guarantee"}
    assert {
        tuple(response["response_alternatives"]) for response in batch
    } == {("\n\nCAP theorem states that a distributed system can only",)}


def test_kimi_linear_basic_accepts_only_observed_h20_cap_output():
    task_info = json.loads(
        (
            RTP_LLM_DIR
            / "test/smoke/data/model/kimi_linear/q_r_bf16_tp2.json"
        ).read_text()
    )
    batch = task_info["query_result"][1]["result"]["response_batch"]

    assert len(batch) == 3
    assert {response["response"] for response in batch} == {
        "\n\nCAP theorem states that a distributed system can only"
    }
    assert {
        tuple(response["response_alternatives"]) for response in batch
    } == {("\n\nCAP theorem states that any distributed system can guarantee",)}


def test_kimi_linear_pd_accepts_only_observed_h20_cap_output():
    task_info = json.loads(
        (
            RTP_LLM_DIR
            / "test/smoke/data/model/kimi_linear/q_r_bf16_tp2_pd_sep.json"
        ).read_text()
    )
    batch = task_info["query_result"][1]["result"]["response_batch"]

    assert len(batch) == 3
    assert {response["response"] for response in batch} == {
        "\n\nCAP theorem states that a distributed system can only"
    }
    assert {
        tuple(response["response_alternatives"]) for response in batch
    } == {("\n\nCAP theorem states that any distributed system can guarantee",)}


def test_kimi_linear_cudagraph_accepts_only_observed_h20_outputs():
    task_info = json.loads(
        (
            RTP_LLM_DIR
            / "test/smoke/data/model/kimi_linear/q_r_cuda_graph.json"
        ).read_text()
    )
    result = task_info["query_result"][0]["result"]
    observed_task_info = json.loads(
        (
            RTP_LLM_DIR
            / "test/smoke/data/model/kimi_linear/q_r_bf16_tp2_kernel_block_size_64.json"
        ).read_text()
    )
    observed_choices = observed_task_info["query_result"][0]["result"][
        "choices_alternatives"
    ][0]

    assert result["choices_alternatives"] == [observed_choices]
    assert {
        choices[0]["message"]["content"]
        for choices in result["choices_alternatives"]
    } == {
        "当然可以！以下是围绕“英语听力方法”主题提出的几个有深度、启发性的问题，适合用于课堂讨论、自我反思或教学研究：\n\n---\n\n### 一、方法类问题\n1. **你通常如何训练英语听力？你觉得自己最有效的方法是什么？为什么？**\n2. **你是否尝试过“精听”和“泛听”结合的方法？你觉得哪种对你帮助更大？**\n3. **你是否使用过“听写”或“跟读”练习？这些方法对你提升"
    }

    batch = task_info["query_result"][1]["result"]["response_batch"]

    assert len(batch) == 3
    assert {response["response"] for response in batch} == {
        "\n\nCAP theorem states that a distributed system can only"
    }
    assert {
        tuple(response["response_alternatives"]) for response in batch
    } == {("\n\nCAP theorem states that any distributed system can guarantee",)}


def test_deepseek_v2_accepts_only_observed_h20_choices_and_usage_pair():
    comparer_source = (RTP_LLM_DIR / "test/smoke/openai_comparer.py").read_text()
    assert '"result_alternatives"' in comparer_source
    assert "alternative_choices == actual_choices" in comparer_source
    assert "alternative_result.usage == actual_result.usage" in comparer_source
    assert 'set(alternative) != {"choices", "usage"}' in comparer_source

    task_info = json.loads(
        (
            RTP_LLM_DIR
            / "test/smoke/data/model/deepseek_v2/q_r_3090_mla.json"
        ).read_text()
    )
    result = task_info["query_result"][0]["result"]
    alternatives = result["result_alternatives"]

    assert len(alternatives) == 1
    assert set(alternatives[0]) == {"choices", "usage"}
    assert result["usage"] == {
        "prompt_tokens": 17,
        "total_tokens": 202,
        "completion_tokens": 185,
    }
    assert alternatives[0]["usage"] == {
        "prompt_tokens": 17,
        "total_tokens": 203,
        "completion_tokens": 186,
    }
    assert result["choices"] != alternatives[0]["choices"]
    assert alternatives[0]["choices"][0]["message"]["content"].endswith(
        "如何将听力训练与口语、阅读和写作等其他语言技能结合起来，"
        "以达到更好的学习效果？"
    )


def test_qwen3_vl_cp2_golden_accepts_only_observed_h20_choices():
    task_info = json.loads(
        (
            RTP_LLM_DIR / "test/smoke/data/model/qwen_vl/q_r_3_cp2.json"
        ).read_text()
    )
    observed_task_info = json.loads(
        (RTP_LLM_DIR / "test/smoke/data/model/qwen_vl/q_r_3.json").read_text()
    )
    result = task_info["query_result"][0]["result"]
    observed_choices = observed_task_info["query_result"][0]["result"]["choices"]
    primary = result["choices"][0]["message"]["content"]
    alternatives = {
        choices[0]["message"]["content"]
        for choices in result["choices_alternatives"]
    }

    assert result["choices_alternatives"] == [observed_choices]
    assert primary.endswith(
        "The dog is reaching its paw out to gently touch the woman's hand in a"
    )
    assert alternatives == {
        "This is a heartwarming, sun-drenched photograph capturing a tender moment between a woman and her dog on a beach at sunset.\n\n**Key Elements:**\n\n*   **The Subjects:** A woman with long, dark hair, wearing a plaid shirt and dark pants, is sitting on the sand. She is smiling warmly, looking at her dog. Beside her, a large, light-colored Labrador Retriever, wearing a colorful harness, sits attentively, extending its paw to give a"
    }


def test_qwen_loader_detects_qwen3_vl_text_tower_prefix():
    tree = ast.parse((RTP_LLM_DIR / "models/qwen_v2.py").read_text())
    process_meta = _find_method(_find_class(tree, "QWenV2Weight"), "_process_meta")
    source = ast.unparse(process_meta)

    assert "self._contains(weight_keys, 'model.language_model.')" in source
    assert "self.prefix = 'model.language_model.'" in source
    assert "self.model_prefix = ''" in source


def test_xqa_backends_reject_sm120():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/xqa.py").read_text())

    for class_name in ("XQAImpl", "XQADecodeImpl"):
        support_source = ast.unparse(
            _find_method(_find_class(tree, class_name), "support")
        )
        assert "torch.cuda.get_device_capability()[0] == 12" in support_source
        assert "return False" in support_source


def test_missing_py_flashinfer_decode_does_not_break_attention_registration():
    tree = ast.parse((ATTENTION_DIR / "__init__.py").read_text())
    source = ast.unparse(tree)

    assert "except ImportError as e:" in source
    assert "PyFlashinferDecodeImpl = None" in source
    assert "if PyFlashinferDecodeImpl is not None:" in source
    assert "DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)" in source


def test_py_flashinfer_module_exports_all_registered_implementations():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    export = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )

    assert ast.literal_eval(export.value) == [
        "PyFlashinferPagedPrefillImpl",
        "PyFlashinferHybridPrefillImpl",
        "PyFlashinferPrefillImpl",
        "PyFlashinferDecodeImpl",
    ]


def test_attention_compatibility_names_survive_remote_source_cache_rollout():
    trt_tree = ast.parse((ATTENTION_DIR / "cuda_impl/trt.py").read_text())
    assignments = {
        target.id: ast.unparse(node.value)
        for node in trt_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["TRTPagedMHAImpl"] == "FlashInferTRTLLMFMHAv2PagedPrefillImpl"
    assert assignments["TRTMHAImpl"] == "FlashInferTRTLLMFMHAv2PrefillImpl"

    xqa_tree = ast.parse((ATTENTION_DIR / "cuda_impl/xqa.py").read_text())
    prepare_source = ast.unparse(
        _find_method(_find_class(xqa_tree, "XQAImpl"), "prepare_cuda_graph")
    )
    assert "update_attention_params" in prepare_source
    assert "getattr(self.fmha_impl, 'update', None)" in prepare_source
    assert "update_kv_cache_offset" in prepare_source
