load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def whisper_suites():
    # whisper-large-v3 ASR (audio URL -> transcript). The audio rides the
    # multimodal channel as request-level cross-attention side state
    # (mm_skip_token_expansion); the decoder prompt is the forced
    # special-token sequence. One entry per acceptance architecture:
    # sm90 -> H20_CU13, sm103 -> SM103_CU13, sm12x -> RTX_5000_PRO_CU13.
    smoke_args = "--act_type FP16 --warm_up 0 --seq_size_per_block 4 --reuse_cache 1 --enable_device_cache 1 --concurrency_limit 8"
    native.test_suite(
        name = "smoke_whisper",
        tests = [
            smoke_test(
                name = "whisper_transcribe_sm103",
                task_info = "data/model/whisper/q_r_transcribe.json",
                smoke_args = smoke_args,
                gpu_type = ["SM103_CU13"],
            ),
            smoke_test(
                name = "whisper_transcribe_h20",
                task_info = "data/model/whisper/q_r_transcribe.json",
                smoke_args = smoke_args,
                gpu_type = ["H20_CU13"],
            ),
            smoke_test(
                name = "whisper_transcribe_rtx5000",
                task_info = "data/model/whisper/q_r_transcribe.json",
                smoke_args = smoke_args,
                gpu_type = ["RTX_5000_PRO_CU13"],
            ),
        ],
    )
