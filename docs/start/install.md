# Install RTP-LLM

We provide multiple ways to install RTP-LLM.
* If you need to run **DeepSeek V3/R1**, it is recommended to refer to [DeepSeek V3/R1 Support](../references/deepseek/index.rst) and use Docker to run
* If you need to run **Kimi-K2**, it is recommended to refer to [Kimi-K2 Support](../references/kimi/index.rst) and use Docker to run
* If you need to run **QwenMoE**, it is recommended to refer to [Qwen MoE Support](../references/qwen/index.rst) and use Docker to run


To speed up installation, it is recommended to use pip to install dependencies:

## Method 1: With pip

```bash
pip install --upgrade pip
pip install "rtp_llm>=0.2.0"
```


## Method 2: From source
| os | Python | NVIDIA GPU | AMD | Compile Tools|
| -------| -----| ----| ----|----|
| Linux | 3.10 | Compute Capability 7.0 or higher <br> ✅ RTX20xx<br>  ✅RTX30xx<br>  ✅RTX40xx<br>  ✅V100<br>  ✅T4<br>  ✅A10/A30/A100<br>  ✅L40/L20<br>  ✅H100/H200/H20/H800.. <br> | ✅MI308X | bazelisk |

Note: the default `--config=cuda13` build targets SM80+; V100/T4 (SM70/SM75) need the retained `cuda12_*` compat configs.


```bash
# Use the last release branch
git clone git@github.com:alibaba/rtp-llm.git
cd RTP-LLM

# build RTP-LLM whl target
# --config=cuda13 build target for NVIDIA GPU with cuda13
# --config=rocm build target for AMD
bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=cuda13 --test_output=errors --test_env="LOG_LEVEL=INFO"  --jobs=64

ln  -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py  `pwd`/rtp_llm/cpp/model_rpc/proto/
ln  -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py  `pwd`/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py

```


### Optional SM120 NVFP4 hybrid Decode runtime

The CUDA 13 default FlashInfer wheel does not expose the static b12x controls
or the CUTLASS `profile_ids` argument used by the hybrid executor. For an
isolated SM120 CUDA 13 environment, the official FlashInfer 0.7.3 wheel contains
the adapter's reference implementations from FlashInfer commit
`7fa5c3b909a74c435a5b89d1bec82ce012cd3ca1`.

```bash
python -m pip install 'flashinfer-python[b12x]==0.7.3'
python -m pip check
```

The upstream `b12x` extra requires PyTorch 2.12 or newer and the CUDA 13
CuTe DSL runtime. The stock RTP-LLM CUDA 13 build pins PyTorch 2.11 and the
FlashInfer 0.6.9 fork. The pip command above is for an isolated regression
environment. A service build using that environment needs a native extension
built against its PyTorch and a matching dependency lock. The stock build
uses CUTLASS.

A service migration to PyTorch 2.12 needs matching build inputs in
`deps/http.bzl` (the PyTorch headers/libraries archive),
`arch_config/arch_select.bzl` (the FlashInfer wheel), and the CUDA 13
requirements and lock files. Rebuild PyTorch-linked extensions in that
environment, including flash-attn, deep-gemm, rtp-kernel, fast-safetensors
and fast-hadamard-transform. Installing a different PyTorch with pip alone
does not change Bazel's PyTorch archive. The pinned FlashInfer 0.6.9 fork
also contains the FMHA v2 CUDA Graph CPU-scale fix. A migration to an
upstream wheel must verify or port that fix and run the attention regressions.

The adapter verifies three entry files, then an aggregate digest over 338 Python
sources in `fused_moe/cute_dsl/blackwell_sm12x/`, `cute_dsl/`, `jit/`, and the
root `utils.py`, `api_logging.py`, `artifacts.py`, `compilation_context.py`.
The reference is the official PyPI `flashinfer_python-0.7.3-py3-none-any.whl`
(SHA256 `548a3ea1f9057025ace0a21b298824f8de049eb4b6c5c458f763309509e2ae40`).
The source digest is
`14f7183343168b25899d866914710f7502e9c2ad30196c142e2449c2db738f5b`:
sort paths, then hash each `flashinfer/<relative-path>`, a NUL byte, and its
binary SHA256. This validates the adapter's FlashInfer Python dependency
surface, not the compiler binaries or all of FlashInfer. Keep the deployment
lock for Torch, CUTLASS DSL and Triton; rerun the deployment gate on changes.
A source mismatch selects CUTLASS and records the reason in the startup log.
A verified runtime records its actual package version and source digest.

The pinned public launcher can select micro kernels even though its name
contains `static`. The adapter does not call that launcher or clone any Python
globals. Its repository-owned `_launch_static` calls `_get_static_kernel`
directly and passes the static workspace ABI explicitly. Micro and dynamic
compilers are unreachable from this entry. The GPU regression replaces the
public launcher and micro compiler entries with failures and requires actual
static compiler calls for every integer M from 1 through 16 on both EP ranks.
The source pin covers this private compiler and its kernel; a new upstream ABI
requires an adapter update and the full regression, not just a version bump.

The scale utility accepts **already-swizzled** CUTLASS block16 bytes and
returns a strided MMA view of the same storage; it must not unswizzle first.
See [the conversion contract](https://github.com/flashinfer-ai/flashinfer/blob/7fa5c3b909a74c435a5b89d1bec82ce012cd3ca1/flashinfer/cute_dsl/utils.py#L367).
In the pinned static kernel, [negative route IDs are skipped](https://github.com/flashinfer-ai/flashinfer/blob/7fa5c3b909a74c435a5b89d1bec82ce012cd3ca1/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py#L1749).
The adapter initializes each local output to zero; the regression poisons it
with NaNs to prove the kernel overwrites remote-only and mixed-slot rows.
[The prologue](https://github.com/flashinfer-ai/flashinfer/blob/7fa5c3b909a74c435a5b89d1bec82ce012cd3ca1/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py#L1686)
clears scratch when needed; [finalize](https://github.com/flashinfer-ai/flashinfer/blob/7fa5c3b909a74c435a5b89d1bec82ce012cd3ca1/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py#L3250)
restores routing counters and publishes the clean marker. A model-owned pool
shares scratch only across serialized layers/Graph replays on its Decode
stream. Weight views stay layer-owned. Concurrent launches on that pool are
unsupported; any failed launch requires worker restart before scratch reuse.

Run the mandatory deployment gate in the matching CUDA 13 dependency lock:
`//rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test:nvfp4_hybrid_sm120_regression`.
This explicit suite includes the CPU contract and one-SM120 GPU target; the
GPU target fails if the pinned ABI is missing. It is manual because the stock
0.6.9 lock cannot run the optional 0.7.3 ABI. From the matching source runtime:

```bash
RTP_TEST_B12X_NVFP4_GPU=1 python -m \
  rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.nvfp4_hybrid_executor_test \
  B12xNvfp4GpuTest
```

The 2026-11-21 SM120 run with the official 0.7.3 wheel passed the six CPU
contract tests and both GPU test methods. The latter cover all 32 combinations
of EP rank 0/1 and M1–16 at local E4/K256/N256/topk2, including nonuniform
scales, remote-only/mixed routes, NaN-poisoned outputs and cross-layer Graph
replay. They also cover M32 CUTLASS `(13,37)` at local
E256/K2560/N640/topk10. These are correctness checks, not a model quality or
service performance equivalence claim; deployments must rerun the gate with
their dependency lock.

The 2026-11-23 production-shape run additionally compared b12x and CUTLASS
at local E256/K2560/N640/topk10 for every integer M1–16 on both EP ranks.
All 32 synthetic and 32 captured-route cases passed. The largest relative
L2 error on captured routes was 3.77%. The regression compares b12x against
CUTLASS eager with a 5% relative L2 limit. Separately, it checks two Graph
replays of each backend against that backend's eager BF16 output, bit for bit.
All 128 CUTLASS and 128 b12x replays passed on SM120.

The Qwen4 FP8-to-NVFP4 recipe selects Hybrid automatically for compatible
SM120 TP2/EP2 Decode workers. The static backend requires this recipe.
`decode_b12x_max_rows` accepts 0 through 16 physical input rows.
Set it to 0 to use CUTLASS for all rows. Set `RTP_NVFP4_HYBRID_MOE=0` to
pin the existing CUTLASS executor for comparisons. Omit
`decode_cutlass_profile_ids` to use the runtime's normal tactic selection.
Explicit tactic IDs are specific to the FlashInfer build and GPU shape, so
validate them on the deployment wheel.

The x86 CUDA 13 native extension includes the SM120 TP2 shared-expert LSA
fusion. Both Decode ranks check the packaged NCCL API, model layout, CUDA
peer access and free memory. They allocate their local window storage
before entering communicator initialization. After initialization, both
ranks agree on NCCL's device LSA capabilities before registering the
symmetric window. A failed preflight, local window allocation or capability
query selects the standard routed collective on both ranks. Both ranks
confirm activation before installing the epilogue. After both NCCL activation
calls return, an unsuccessful local capability check also sends both ranks
back to the standard collective. Failures inside NCCL
communicator initialization, symmetric-window registration or device
communicator creation can leave a peer in a native collective and require
the TP group to restart.
Both ranks use the same logical batch shape to select the epilogue and size
its grid. The grid has at most 64 CTAs. Each CTA uses its block index as
one of the 64 reserved barrier slots, and the final barrier protects the
symmetric input window until both peers have finished reading it.
The acquire/release barriers publish the staged partial before peer reads
and retire peer reads before the next layer reuses the window. NCCL 2.30.4
stores [an epoch per barrier index](https://github.com/NVIDIA/nccl/blob/v2.30.4-1/src/include/nccl_device/impl/lsa_barrier__funcs.h#L30).
Unused reserved slots do not participate. Both peers must use identical
logical M and H, hence identical grids; every CTA index is checked to fit the
64-slot reservation. Activation checks the actual
`dev.lsaBarrier.nBarriers >= 64`, rather than relying on the requested count,
and requires a non-null device abort flag. [NCCL 2.30.4's initializer](https://github.com/NVIDIA/nccl/blob/v2.30.4-1/src/dev_runtime.cc#L1318)
sets that flag only for its internal communicators. Before bootstrap, the
extension allocates a device-local cancellation word, one pinned host word
and a nonblocking copy stream. Normal checks read device memory. Fatal
cancellation copies the pinned constant through the independent stream,
which requires an asynchronous copy engine. The owner retains these resources
for the lifetime of captured Graphs. This is an ordinary CUDA launch, as in NCCL's
[local LSA AllReduce example](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#local-allreduce-using-lsa).

Both FP8-to-NVFP4 and prequantized NVFP4 checkpoints accept the Decode LSA
options in the ModelOpt FP4 configuration. For a prequantized checkpoint,
use `--quantization` with the existing packed format and the desired options:

```json
{"method":"modelopt_fp4","bits":4,"group_size":16,"is_quanted":true,"decode_lsa_enabled":false}
```

`decode_lsa_enabled` defaults to true. A rank that opts out reports that
decision in the layout exchange, so both ranks retain the standard collective.
LSA uses the main TP group's 36,500-day slow-start policy for each host
exchange during layout, allocation, capability and readiness checks. To bound
each exchange, set `decode_lsa_startup_timeout_s` to the same positive number
of seconds on both ranks. A store timeout fails startup for the TP group.
MTP draft models skip the LSA exchanges. The two-rank SM120 regression
delays rank 1 by 123 seconds before the layout exchange, then verifies native
LSA activation and output after the peer arrives.

`decode_lsa_timeout_ms` in the ModelOpt FP4 config defaults to 0: wait on the
owner's cancellation flag without a device deadline. Both ranks agree
on the value before creating the communicator. A positive value opts into
NCCL's timed barrier and a fatal CUDA trap. Its cycles are milliseconds times
`cudaDevAttrClockRate` (peak kHz); NCCL uses `clock64()`, so clock throttling
can lengthen the wall time. It is a nominal device deadline, not a precise
wall-clock watchdog. Leave it at 0 to use the service's normal TP supervisor
policy; choose a positive deadline only with a matching worker restart policy.
[The NCCL polling implementation](https://github.com/NVIDIA/nccl/blob/v2.30.4-1/src/include/nccl_device/impl/lsa_barrier__funcs.h#L94)
contains both abort-driven and timed waits. Its untimed API returns `void`;
it does not propagate an abort status. The fusion explicitly checks the device
abort flag after each barrier and traps before reading the peer window or
allowing a later staged copy to reuse it. An absent peer does not set this private
communicator's local abort flag. With timeout 0, a process supervisor must still
terminate/restart both TP workers on a peer failure; there is no promise of an
automatic device deadline. `ops.abort` first queues the device cancellation
flag write and then calls `ncclCommAbort`. It is fatal cancellation for a live local
communicator, not a fallback to another collective. The flag is not an alias
of NCCL's private host communicator state.

ARM keeps the standard collective: the SM120 native target and binding are
currently enabled only for x86 CUDA 13. Required `nccl.h` and `nccl_device.h`
are explicit build inputs, and the supporting header glob must be nonempty.
Build the complete x86 SM120 Python compute extension, including the registered
LSA ops, with `bazel build --config=cuda13 --config=sm120 //:rtp_compute_ops`.

Aligned staging buffers are allocated during group-agreed initialization;
each output is allocated before suppressing the routed collective, with
independent ownership for each forward/Graph node. A local failure after
suppressing that collective is fatal to the TP group: the supervisor must
restart both workers. Never switch one rank to a different collective or
reuse a failed CUDA context. Model close retains native contexts until
process exit while captured Graph addresses may still reference them.

Run `//rtp_llm/models_py/modules/qwen4_exp/test:reduce_gate_integration_test`
on CUDA 13. It reserves two SM120 GPUs and is included in `qwen4_moe`.
It injects one-rank allocation/capability failures and shared-weight packing
failures, verifies delayed arrival, and compares M1–32 across H8–8192 in
repeated eager/Graph calls. Cross-implementation CUDA-versus-NCCL/Triton
comparisons use the existing BF16 gating tolerance (`atol=rtol=1e-2`) and
record unequal-element counts and maximum absolute error. Eager-versus-Graph
of the same fused implementation remains byte-identical. The source invocation is:

```bash
NCCL_P2P_DISABLE=0 python -m \
  rtp_llm.models_py.modules.qwen4_exp.test.moe.reduce_gate_integration_test
```

The manual `reduce_gate_missing_peer_test` opts into a nominal 1-second
deadline, makes rank 1 omit the collective, checks rank 0's CUDA fault and
wall-clock bounds, and exits both workers without reusing the faulted context.
`reduce_gate_default_abort_test` is included in `qwen4_moe`: with timeout 0,
rank 1 omits its launch and rank 0 explicitly aborts its local communicator
while all 64 barrier CTAs wait;
the queued fusion must fault rather than complete with stale peer data.
Both tests run in disposable processes; a hung cancellation fails the bounded
test process rather than attempting to recover its CUDA context.
`reduce_gate_peer_exit_test` additionally exits one worker while the peer has
an unmatched launch in flight. The surviving worker must report a CUDA fault
or remain pending until the test supervisor orders local cancellation. The
test supervisor reaps both workers within the test budget.
`reduce_gate_tp_recovery_test` runs the same unmatched native launch under
RTP-LLM's backend `ProcessManager`. One rank exits with status 42 while the
other waits and does not respond to graceful shutdown signals. The manager
escalates to SIGKILL, reaps the surviving rank and exits with status 1 for
the deployment supervisor. The test launches a fresh TP
group with new processes and a new communicator, and checks the resulting
eager output and two CUDA Graph replays on both ranks. The test controls the
restart so it can run outside a deployment orchestrator.

The 2026-11-21 CUDA 13.2/NCCL 2.30.4 run on two SM120 GPUs passed all 36
shape cases, both one-rank startup failures, all eight shared-packing agreement
cases, delayed arrival, the timed missing-peer test and the default cancellation
test with 64 waiting CTAs. The recorded cross-implementation comparisons had
zero unequal elements in this run; their portable acceptance contract remains
the BF16 tolerance above. Same-implementation eager/Graph checks were bitwise.

On the admitted BF16 Decode path, the shared projections and gate run on
an auxiliary CUDA stream and join the routed result before the LSA epilogue.
For a static Qwen4 checkpoint, set `"decode_shared_gemm_nk": true` in the
ModelOpt FP4 quantization config to also pack contiguous BF16 shared
projection weights. The measured shapes use packed weights for up at
physical M1–16 and down at M2–32. A 48-layer Qwen3.8-Flash-Next worker
uses an additional 450 MiB per rank. This option defaults to false.
Both ranks preflight the layout and memory, then build private copies and
agree again before replacing modules or marking source weights immutable.
If either fails, both release private copies, retain the original projections
and keep their original update policy. Updating successfully packed weights
requires restarting the model and rebuilding the CUDA Graphs.

FP8 Dense keeps its existing shared-expert overlap qualification for both
CUTLASS and Hybrid routed executors. BF16 Dense uses the separate model-owned
LSA qualification above. Unsupported model layouts retain their original
collective and shared computation path.

## Method 3: Using docker
More Docker versions can be obtained from [RTP-LLM Release](../release/index.rst)
```bash
docker run --gpus all \
 --shm-size 32g \
 -p 30000:30000 \
 -v /mnt:/mnt \
 -v /home:/home \
 --ipc=host \
ali-hangzhou-hub-registry.cn-hangzhou.cr.aliyuncs.com/isearch/rtp_llm_sm8x_opensource:0.2.0_0.2.0_2025_10_09_17_35_8fa289f5 \
  /opt/conda310/bin/python -m rtp_llm.start_server \
   --checkpoint_path=/mnt/nas1/hf/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/6114e9c18dac0042fa90925f03b046734369472f/ \
    --model_type=qwen_2 --start_port=30000

```

## Method 4: Using Kubernetes
This guide walk you through deploying the RTP-LLM service on Kubernetes. You can deploy RTP-LLM to Kubernetes using any of the following:

- [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [StatefulSet](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [LWS](https://lws.sigs.k8s.io/docs/overview/)

### Deploy with Kubernetes Deployment
You can use a native Kubernetes Deployment to run a single-instance model service.

1. Create the deployment resource to run the RTP-LLM server. Example:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: Qwen1.5-0.5B-Chat
  namespace: default
  labels:
    app: Qwen1.5-0.5B-Chat
spec:
  replicas: 1
  selector:
    matchLabels:
      app: Qwen1.5-0.5B-Chat
  template:
    metadata:
      labels:
        app: Qwen1.5-0.5B-Chat
    containers:
    - name: rtp-llm
      image:ali-hangzhou-hub-registry.cn-hangzhou.cr.aliyuncs.com/isearch/rtp_llm_sm8x_opensource:0.2.0_0.2.0_2025_10_09_17_35_8fa289f5
      command:
      - /opt/conda310/bin/python
      - -m
      - rtp_llm.start_server
      - --checkpoint_path
      - /mnt/nas1/hf/models--Qwen--Qwen1.5-0.5B-Chat/
      - --model_type
      - qwen_2
      - --start_port
      - "30000"
    resources:
      requests:
        cpu: "2"
        memory: 6G
        nvidia.com/gpu: "1"
      limits:
        cpu: "10"
        memory: 20G
        nvidia.com/gpu: "1"
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
    livenessProbe:
      httpGet:
      path: /health
      port: 30000
      initialDelaySeconds: 60
      periodSeconds: 10
    readinessProbe:
      httpGet:
      path: /health
      port: 30000
      initialDelaySeconds: 60
      periodSeconds: 5
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: "2Gi"
```

2. Create a Kubernetes Service to expose the RTP-LLM server
```yaml
apiVersion: v1
kind: Service
metadata:
  name: Qwen1.5-0.5B-Chat
  namespace: default
spec:
  type: ClusterIP
  ports:
  - name: server
    port: 80
    protocol: TCP
    targetPort: 30000
  selector:
    app: Qwen1.5-0.5B-Chat
```

3. Deploy and Test

Apply the deployment and service resources using `kubectl`.
```shell
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```
Send a request to verify the model service is working properly.
```shell
curl -X POST http://Qwen1.5-0.5B-Chat.svc.cluster.local/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "default",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant"
    },
    {
      "role": "user",
      "content": "你是谁？"
    }
  ],
  "temperature": 0.6,
  "max_tokens": 1024
}'
```

### Multi-Node Deployment
When deploying a large-scale model, you may need multiple pods to deploy a single model service instance. The native Kubernetes Deployments and StatefulSets cannot manage multiple pods as a single unit throughout their lifecycle. In this case, you can use the community‑maintained LWS resource to handle the deployment.

As an example, to deploy the Qwen3‑Coder‑480B‑A35B‑Instruct model with tp=8, request two pods with 4 GPUs each. The lws deployment yaml is as follows:

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: Qwen3-Coder-480B-A35B-Instruct
  namespace: default
  labels:
    app: Qwen3-Coder-480B-A35B-Instruct
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          app: Qwen3-Coder-480B-A35B-Instruct
          role: leader
      spec:
        containers:
        - name: rtp-llm
          image:ali-hangzhou-hub-registry.cn-hangzhou.cr.aliyuncs.com/isearch/rtp_llm_sm8x_opensource:0.2.0_0.2.0_2025_10_09_17_35_8fa289f5
          command:
          - python3
          - -m
          - rtp_llm.start_server
          - --checkpoint_path
          - /mnt/nas1/hf/Qwen__Qwen3-Coder-480B-A35B-Instruct/
          - --model_type
          - qwen3_coder_moe
          - --start_port
          - "30000"
          - --tp_size
          - "8"
          - --world_size
          - $(LWS_GROUP_SIZE)
          - --world_index
          - $(LWS_WORKER_INDEX)
          - --leader_address
          - $(LWS_LEADER_ADDRESS)
          resources:
            limits:
              cpu: "96"
              memory: 800G
              nvidia.com/gpu: "4"
          volumeMounts:
          - name: shm
            mountPath: /dev/shm
          livenessProbe:
            httpGet:
            path: /health
            port: 30000
            initialDelaySeconds: 60
            periodSeconds: 10
          readinessProbe:
            httpGet:
            path: /health
            port: 30000
            initialDelaySeconds: 60
            periodSeconds: 5
        volumes:
        - name: shm
            emptyDir:
            medium: Memory
    workerTemplate:
      metadata:
        labels:
          app: Qwen3-Coder-480B-A35B-Instruct
          role: worker
      spec:
        containers:
        - name: rtp-llm
          image:ali-hangzhou-hub-registry.cn-hangzhou.cr.aliyuncs.com/isearch/rtp_llm_sm8x_opensource:0.2.0_0.2.0_2025_10_09_17_35_8fa289f5
          command:
          - python3
          - -m
          - rtp_llm.start_server
          - --checkpoint_path
          - /mnt/nas1/hf/Qwen__Qwen3-Coder-480B-A35B-Instruct/
          - --model_type
          - qwen3_coder_moe
          - --start_port
          - "30000"
          - --tp_size
          - "8"
          - --world_size
          - $(LWS_GROUP_SIZE)
          - --world_index
          - $(LWS_WORKER_INDEX)
          - --leader_address
          - $(LWS_LEADER_ADDRESS)
          resources:
            limits:
              cpu: "96"
              memory: 800G
              nvidia.com/gpu: "4"
          volumeMounts:
          - name: shm
            mountPath: /dev/shm
          livenessProbe:
            httpGet:
            path: /health
            port: 30000
            initialDelaySeconds: 60
            periodSeconds: 10
          readinessProbe:
            httpGet:
            path: /health
            port: 30000
            initialDelaySeconds: 60
            periodSeconds: 5
        volumes:
        - name: shm
            emptyDir:
            medium: Memory
```
