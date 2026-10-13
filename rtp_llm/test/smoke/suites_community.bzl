load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")


"""Community-adapted model smoke suites.

Every case here serves a checkpoint adapted by the model-adaptation flow
(decision models, community releases) through the decision schema
(/v1/classifier: document + typed questions -> per-question
probabilities).  Fixtures live under data/model/community/ and reference
the shared 3fs models volume or staged copies; update the model_path in
the fixture when a checkpoint moves.

These cases are OFF by default: the suite carries the `manual` tag and is
not wired into any CI lane.  New model adaptations add their smoke case
HERE (not to the platform suites) -- promote a case to a platform suite
only after it is green on the target GPU pool and the checkpoint has a
CI-reachable home.  Run locally with:

    bazel test //rtp_llm/test/smoke:<case> --config=cuda13
"""


def community_suites():
    # Decision models: every case exercises the /v1/classifier decision
    # schema (document + typed questions -> per-question probabilities)
    # through the unified decision module.  Serving copies live on the
    # 3fs volume (original releases served in place where the release
    # layout allows it; JEV currently points at a LoRA-merged serving copy
    # with the golden taken from the merged-checkpoint reference -- pass
    # --lora_info against the original release instead once the original
    # layout is staged for CI).
    native.test_suite(
        name = "smoke_community_decision_models",
        tags = ["manual"],
        tests = [
            smoke_test(
                name = "community_kev_05b_decision",
                task_info = "data/model/community/kev_05b_decision.json",
                smoke_args = "--task_type SEQ_CLASSIFICATION --act_type BF16 --max_seq_len 2048 --warm_up 0",
                gpu_type = ["SM103_CU13"],
            ),
            smoke_test(
                name = "community_autojev_27b_decision",
                task_info = "data/model/community/autojev_27b_decision.json",
                smoke_args = "--task_type SEQ_CLASSIFICATION --act_type BF16 --max_seq_len 8192 --warm_up 0",
                gpu_type = ["SM103_CU13"],
            ),
            smoke_test(
                name = "community_jev_27b_vl_decision",
                task_info = "data/model/community/jev_27b_vl_decision.json",
                smoke_args = "--task_type SEQ_CLASSIFICATION --act_type BF16 --max_seq_len 8192 --warm_up 0",
                gpu_type = ["SM103_CU13"],
            ),
            smoke_test(
                name = "community_clef_27b_decision",
                task_info = "data/model/community/clef_27b_decision.json",
                smoke_args = "--task_type SEQ_CLASSIFICATION --act_type BF16 --max_seq_len 8192 --warm_up 0",
                gpu_type = ["SM103_CU13"],
            ),
            smoke_test(
                name = "community_decider_35b_a3b_decision",
                task_info = "data/model/community/decider_35b_a3b_decision.json",
                # 65GB bf16 weights + KV cache do not fit a single 72GB card.
                smoke_args = "--task_type SEQ_CLASSIFICATION --act_type BF16 --tp_size 2 --seq_size_per_block 2048 --max_seq_len 4096 --warm_up 0",
                gpu_type = ["SM103_CU13"],
            ),
        ],
    )
