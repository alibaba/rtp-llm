package(default_visibility = ["//visibility:public"])

# Runtime closure for the standalone priority-preemption smoke. Proto sources
# and the FlexLB jar are supplied by the consuming sh_test because they belong
# to the enclosing RTP-LLM build, not to this Python tool package.
filegroup(
    name = "priority_smoke_runtime",
    srcs = [
        "flexlb_smoke_base.py",
        "mock_engine_cluster.py",
        "priority_preemption_smoke.py",
        "run_cancel_smoke.sh",
    ] + glob(["online_eval/*.py"]),
)
