# Opt-in ROCm Qwen4 MTP hyper-connection fusion

This patch enables ATOM's existing fused HC path inside the MTP drafter,
materializes the final pending residual before returning its hidden states,
and uses the fused HC head-input mixer. It does not change the target model,
GDN state handling, KV layout, acceptance algorithm or sampling configuration.
The switch defaults off and is ignored on non-ROCm builds. Enable it only on
the validated D role; P keeps its baseline configuration.

## Install after the ATOM wheel and existing dependency patches

Requires ATOM `0.1.8+g9af739efc` (wheel SHA256 in `manifest.json`), the
existing RTP/ATOM adapters, amd-aiter `0.1.29+rocm7.2.0.git97d2389` and FlyDSL
0.4.2. Do not apply to the older 0.1.1 wheel. This directory is an optional
post-wheel patch, not a change to the default image dependency installation.
Use the Python interpreter that actually runs the service:

```sh
python3 patches/atom/mtp_hc/apply.py
python3 patches/atom/mtp_hc/apply.py --verify-only
```

For the internal copy use `internal_source/dockerfile/mtp_hc/apply.py`.
The installer checks version and source/patch/output hashes, validates in a
temporary directory, refuses unknown local edits, and is idempotent. Restart
workers after applying. There is no C++ change required by this patch.

Using the committed service profile (run each role in its own terminal):

```sh
export MODEL_DIR=/local_model_root/qwen4-125b-qsa-taobao-v16-ptpc-fp8
export WORK_DIR=/home/xingjunna.xjn/xjn_116_debug/hc_repro
export RTP_PYTHON=/home/xingjunna.xjn/xjn_116_debug/repro/venv/bin/python
ATOM_RTP_MTP_HC_FUSION=0 bash rtp_llm/test/perf_test/rocm_pd_qwen4/serve_pd.sh prefill
ATOM_RTP_MTP_HC_FUSION=1 bash rtp_llm/test/perf_test/rocm_pd_qwen4/serve_pd.sh decode
```

Use `ATOM_RTP_MTP_HC_FUSION=0` for the D baseline. P uses GPUs0/1, D2/3,
TP=EP=2 each, gen3, async stream and prepare enabled, post-rejection prepare
 disabled, P reuse on and D reuse off. Preserve the existing AITER/JIT/ABI
environment of the installed RTP build. The exact tested launcher and full
benchmark harness are archived on 88_amd below.

## Validation on 88_amd / xjn_112_test

Clean rebuilt RTP sources: external `8f9a9a1220a4aebd10efbb10aa35825478b24f8d`,
internal `d3f65aea6f354adff943d57b394e137c432ba5d7`.
The ineffective host-metadata D2H candidate was removed before this build.
Both modes passed all 36 baseline questions and all 36 feature checks.
The HC primitive smoke test for batch1/2, H2560/HC4/rank320 returned exactly
matching BF16 mixed outputs and raw injection values for its seeded inputs;
this is limited test coverage, not a universal bitwise-equivalence guarantee.

Each performance round has 20 sequential requests after warmup. Input6656,
output347, actual P cache hit5120, concurrency1. All120 timed requests passed
length, hit and PD assertions. TTFT/TPOT are client timings without profiling.
TPOT=(end-to-end time - TTFT)/(347-1). Acceptance is the mean per-request
accepted draft tokens / (draft rounds * 3), not artificially fixed at60%.

| Round | Base TTFT ms | HC TTFT ms | Base TPOT ms | HC TPOT ms | Base acceptance | HC acceptance |
|---|---:|---:|---:|---:|---:|---:|
|1|286.336|287.969|4.9315|4.8216|59.269%|60.065%|
|2|291.365|285.095|4.8979|4.7683|59.887%|61.160%|
|3|284.431|286.220|4.8469|4.8069|61.026%|60.618%|
|Mean|287.377|286.428|4.8921|4.7989|60.061%|60.614%|

Mean TPOT decreased1.90%; all three round means improved. This does **not**
meet the4.74ms target. Acceptance increased0.554 percentage points and runs
were sequential rather than randomized, so not all improvement can be
attributed to HC kernel cost alone. Broader workload and tail-latency gains
are not established. Retain the opt-in switch.

After three warmup inferences, the D trace contains174 `hc_pre` and174
`hc_mix` calls; `mix_gated_mean` has zero calls. Thus the fused path really
ran. The profiled request is excluded from performance statistics. Kernel
counts from separate traces must not be treated as normalized speedups.

Artifacts on88_amd:

- `/home/xingjunna.xjn/xjn_116_debug/pd_mtp_hc_fusion/`: clean build script/log,
  wheel/hash, candidate diff, HC smoke test/log, capture script/result.
- `/home/xingjunna.xjn/xjn_116_debug/pd_decode_head_sync_opt/start_async.sh`:
  exact tested launcher (set the switch on the D process before launching).
- Same directory: `run_validation.py`, `logic36.py`, `baseline_benchmark.py`;
  run `python run_validation.py hc_clean_base` or `hc_candidate` after starting
  the corresponding mode. These files preserve the exact seeds and requests.
- Same directory: `hc_clean_base_logic.json`, `hc_candidate_logic.json` and
  `{hc_clean_base,hc_candidate}_round{1,2,3}/result.json`.
- Trace:
  `/home/xingjunna.xjn/xjn_116_debug/pd_decode_head_sync_opt/decode/intended_run/hc_fused_decode_wr0_1.json`
