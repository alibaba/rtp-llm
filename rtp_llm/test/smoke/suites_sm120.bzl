load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm120_suites():
    # SM120 / RTX 5000 Pro / Blackwell consumer (GB202)
    # ============================================================================
    # 硬件约束（写死规则，决定哪些 case 可以放进来）：
    #   - 单卡 32GB VRAM；TP=2 走 PCIe，DeepEP-LL 因无 NVLink 不可用
    #   - sm_120a CUDA arch；FlashInfer v0.6.12rc1+rtp.260523 已带 b12x AOT
    #   - cuda12_9 工具链
    # 维护规则：
    #   - 每个 PR-N 跑通才把对应 case 落到这里（避免 CI 噪声）
    #   - 命名 `<case>_sm120` 后缀，避免和 sm8x/h20 同名 py_test 冲突
    #   - mainse_bert 等内源协议 case 放在 internal_source/.../suites_sm120_internal.bzl
    # ============================================================================

    # SM120 Basic (sm8x_basic 同位 — Qwen2.5 dense FP16/BF16)
    # 对应 PR-4: Qwen dense FP16/BF16
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            smoke_test(
                name="bf16_sm120",
                task_info="data/model/qwen25/q_r_s.json",
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # TODO(PR-4): fp16_sm120 / tp2_sm120 / softmax_probs_sm120 等
            # 等 PR-4 实跑通过后再加 case + 录 golden
        ],
    )


    # SM120 Dense (h20_dense 同位 — Qwen3 dense FP8)
    # 对应 PR-5: Qwen dense FP8 PER_BLOCK
    # TODO(PR-5): dense_fp8pb_sm120 / dense_fp8pt_sm120 等
    # 暂时占位，等 DeepGEMM 在 sm_120 上验证后再填


    # SM120 MoE (h20_moe + sm100_moe 同位 — Qwen3-30B MoE)
    # 对应 PR-6 (FP8) + PR-9 (NVFP4 可选)
    # 注意：RTX 5000 Pro 无 NVLink → DeepEP 路径全部不在此 suite 内
    #       仅跑 fp8_per_block_no_dp_masked / pure_dp / nvfp4 non-deepep
    # TODO(PR-6): moe_masked_fp8_sm120
    # TODO(PR-9): moe_nvfp4_no_deepep_sm120


    # SM120 Next (h20_next 同位 — Qwen3-Next BF16/FP8 + linear attention)
    # 对应 PR-7: Qwen3-Next + linear attention
    # TODO(PR-7): next_bf16_basic_sm120 / next_fp8_basic_sm120
