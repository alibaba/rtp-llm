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
    # 跨架构 golden 注意:
    #   - softmax_probs / fp16 / bf16 用 max_new_tokens<=10 + top_k=1 greedy，
    #     golden 在 L20/H20/SM_120 上稳定，可以直接复用 L20 task_info
    #   - random_seed (top_k=100 采样) / frontend_app (frontend+pd_fusion 双角色)
    #     在 sm_120 上各自踩到不同 blocker，参见下方 TODO 注释
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            smoke_test(
                name="softmax_probs_sm120",
                # sm120-specific golden: FMHA on sm_120a uses a different code
                # path than L20, producing ~1e-3 numerical drift on softmax
                # probabilities. fp16/bf16 above use top_k=1 greedy and stay
                # byte-identical to L20 golden, so they share q_r_s.json.
                task_info="data/model/qwen25/q_r_softmax_probs_sm120.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="fp16_sm120",
                task_info="data/model/qwen25/q_r_s.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="bf16_sm120",
                task_info="data/model/qwen25/q_r_s.json",
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # TODO(PR-4 followup): random_seed_sm120
            #   B-5 已通过本 PR 在 trtllm_gen.py / xqa.py support() 加 sm_120
            #   短路 fallback 到 PyFlashinferPaged 解决；理论上 top_k=100 采样
            #   decode 现在能跑通。仍需独立 PR：(1) 在 sm_120 上跑 rewrite_smoke
            #   录 golden  (2) 比对 L20 golden 看 random seed sampler 是否
            #   bit-equivalent  (3) 落 case。
            # TODO(PR-4 followup): frontend_app_sm120
            #   blocker B-6: smoke_args dict 同时声明 frontend + pd_fusion
            #   两个 role，gpu_count=2，本机单卡 RTX 5000 Pro 起不来。
            #   CI sm12x stage 是 2 卡 → 这个 case 需要去 CI 实测一次再落 case。
            # TODO(PR-4 followup): tp2_sm120 / beam_search_tp2_sm120
            #   单卡 RTX 5000 Pro 跑不了；需要 2 卡环境（无 NVLink，PCIe TP）
            # TODO(PR-4 followup): embedding_qwen_gte_7b_cudagraph_sm120
            #   依赖 gte-Qwen2-7B (~14GB) 模型下载 + CUDA Graph 独立验证
            # TODO(PR-4): qwen3_1_7b_fp16_sm120 / qwen3_1_7b_bf16_sm120
            #   待 Qwen3-1.7B 下载到 /mnt/nas1/hf/ 后落地
        ],
    )


    # SM120 Dense (h20_dense 同位 — Qwen3 dense FP8)
    # 对应 PR-5: Qwen dense FP8 PER_BLOCK
    # TODO(PR-5): dense_fp8pb_sm120 / dense_fp8pt_sm120 等
    # 暂时占位，等 DeepGEMM 在 sm_120 上验证后再填

    # SM120 Next (h20_next 同位 — Qwen3-Next BF16/FP8 + linear attention)
    # 对应 PR-7: Qwen3-Next + linear attention
    # TODO(PR-7): next_bf16_basic_sm120 / next_fp8_basic_sm120
