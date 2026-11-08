"""A concrete test model registration around the real native P/D lifecycle.

Only the distinct test model observes forward inputs/outputs. Canonical model
selection and the C++-imported public RuntimeExpertBalancer are unchanged.
"""

import json
import os
import sys
from pathlib import Path

MODEL_TYPE = "deepseek_v41_ppu_shared_eplb_smoke"
MODULE = "rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_eplb_role"

if os.environ.get("RTP_V41_N8_EPLB_FIXTURE") == "1":
    # Re-executed by multiprocessing spawn, as in shared_server.py. This is
    # ordinary lazy model registration, not a replacement of any factory.
    from rtp_llm.model_factory_register import register_lazy_model

    register_lazy_model(MODEL_TYPE, MODULE)

    if __name__ == MODULE:
        from rtp_llm.model_factory_register import register_model
        from rtp_llm.models.deepseek_v41 import DeepSeekV41
        from rtp_llm.models_py.modules.dsv41.ppu.shared_model import DeepSeekV41SharedPPUModel
        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_eplb import (
            NativeCallObserver,
            NativeEplbEvidence,
            record_prefill_lineage,
        )

        class NativeSharedEplbModel(DeepSeekV41SharedPPUModel):
            def forward(self, inputs, fmha_impl=None):
                if os.environ.get("RTP_V41_TEST_EPLB_GRAPH") == "1":
                    return super().forward(inputs, fmha_impl)
                evidence = getattr(self, "_n8_evidence", None)
                if evidence is None:
                    evidence = self._n8_evidence = (
                        NativeEplbEvidence(self) if self._is_decode_role else NativeCallObserver(self)
                    )
                if evidence.done:
                    # The native stop vote can let ranks exit independently.
                    # Once both requests are observed, leave only the real forward.
                    return super().forward(inputs, fmha_impl)
                evidence.before(inputs)
                output = super().forward(inputs, fmha_impl)
                observation = evidence.after(inputs, output)
                if not self._is_decode_role and observation[0]["kind"] == "business":
                    record_prefill_lineage(self, inputs, output)
                    if observation[0]["stage"] == "fresh":
                        evidence.finalize()
                return output

            def prepare_fmha_impl(self, inputs, is_cuda_graph=False, **kwargs):
                result = super().prepare_fmha_impl(inputs, is_cuda_graph=is_cuda_graph, **kwargs)
                if is_cuda_graph and result is not None and os.environ.get("RTP_V41_TEST_EPLB_GRAPH") == "1":
                    self._eplb_graph_captures = getattr(self, "_eplb_graph_captures", 0) + 1
                return result

            def prepare_native_graph_step(self, inputs):
                ready = super().prepare_native_graph_step(inputs)
                if ready and not inputs.warmup and os.environ.get("RTP_V41_TEST_EPLB_GRAPH") == "1":
                    from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_eplb_graph import ReplayEvidence

                    evidence = getattr(self, "_eplb_replay_evidence", None)
                    if evidence is None:
                        evidence = self._eplb_replay_evidence = ReplayEvidence(self, self._eplb_graph_captures)
                    evidence.before(inputs, self._eplb_graph_captures)
                return ready

            def note_native_graph_replay(self):
                super().note_native_graph_replay()
                evidence = getattr(self, "_eplb_replay_evidence", None)
                if evidence is not None and evidence.pending is not None:
                    evidence.after()

        class NativeSharedEplb(DeepSeekV41):
            def _create_python_model(self):
                self.py_model = NativeSharedEplbModel(
                    self.model_config, self.parallelism_config, self.weight, self.moe_config,
                    max_generate_batch_size=self.max_generate_batch_size,
                    fmha_config=self.fmha_config, py_hw_kernel_config=self.hw_kernel_config,
                    device_resource_config=self.device_resource_config,
                )

        register_model(MODEL_TYPE, NativeSharedEplb)


def server_main():
    from rtp_llm.tools.v41.launch import preset_env

    if os.environ.get("RTP_V41_N8_EPLB_FIXTURE") != "1":
        raise ValueError("N8 test bootstrap requires its explicit fixture selector")
    role = os.environ["RTP_V41_SHARED_ROLE"]
    target = role == "decode"
    graph = os.environ.get("RTP_V41_TEST_EPLB_GRAPH") == "1"
    env = preset_env(
        role, os.environ["RTP_V41_TEST_CHECKPOINT"], rdma=True,
        prefill_address=os.environ["RTP_V41_SHARED_PREFILL_ADDRESS"],
        decode_address=os.environ["RTP_V41_SHARED_DECODE_ADDRESS"],
        start_port=int(os.environ["RTP_V41_SHARED_DECODE_ADDRESS" if target else "RTP_V41_SHARED_PREFILL_ADDRESS"].rsplit(":", 1)[1]),
        ppu_model_path="shared", kv_cache_sharded=True, ced_mode="bounded",
        yoco_keep_tokens=140, max_rollback_tokens=0, max_seq_len=1024 if graph else 512,
        fake_balance_expert=False, shared_eplb=True, redundant_experts=8,
        # Short native-round cadence; a two-sample public window retains
        # real counts across one ordinary all-empty round. Never set steps.
        eplb_mode="EPLB" if target else "NONE", eplb_update_time=2 if target else 0,
        eplb_stats_window_size=2 if target else 1,
        shared_grouped_graph=target and graph,
    )
    # These are immutable startup options. No live config/count/map mutation.
    env.update(MODEL_TYPE=MODEL_TYPE,
               EPLB_METRICS_REPORT_INTERVAL="4", EPLB_METRICS_SNAPSHOT_MAX_FILES="1",
               ENABLE_CUDA_GRAPH="1" if target and graph else "0", RTP_LLM_STREAM_ASYNC="0")
    if graph:
        env["CONCURRENCY_LIMIT"] = "4" if target else "32"
        if target:
            env["DECODE_CAPTURE_CONFIG"] = "1,2,4"
    os.environ.update(env)
    sys.argv[1:] = []
    from rtp_llm.start_server import main

    main()


def main():
    if sys.argv[1:] == ["--server"]:
        return server_main()
    if sys.argv[1:]:
        raise ValueError("N8 bounded native role accepts no matrix overrides")
    from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_role import main as native_role

    native_role(sharded=True, eplb=True)
    if os.environ.get("RTP_V41_TEST_EPLB_GRAPH") == "1":
        return
    output = Path(os.environ["RTP_V41_SHARED_OUTPUT"])
    if os.environ["RTP_V41_SHARED_ROLE"] == "decode":
        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_eplb import validate_native_reports

        summary = validate_native_reports(output / "eplb")
        summary.update(SMOKE="PASS_NATIVE_EPLB_SAME_INSTANCE", FULL_VALIDATION="DEFERRED_TO_N10")
        (output / "eplb-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_eplb import validate_prefill_lineage

        summary = validate_prefill_lineage(output / "eplb")
        (output / "eplb-http-lineage.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
