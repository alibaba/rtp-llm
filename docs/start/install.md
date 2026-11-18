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

The adapter checks the verified FlashInfer dispatcher, static kernel and scale
utility sources and checks the launcher's direct backend flag binding. The
workspace and launcher use the same static backend. The adapter initializes
the local partial to zero. In the pinned kernel, negative IDs are skipped and
every launch resets routing scratch before work on the serialized Decode
stream. This scratch-only contract permits sharing workspace across layers
and successive Graph replays. The scale utility views existing CUTLASS
swizzled bytes in the MMA layout. A wheel with different sources uses CUTLASS.
[FlashInfer 0.7.3 static MoE source](https://github.com/flashinfer-ai/flashinfer/blob/7fa5c3b909a74c435a5b89d1bec82ce012cd3ca1/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py)

Run the b12x regression before deploying the optional wheel. The manual
`nvfp4_hybrid_sm120_gpu_test` Bazel target reserves one SM120 GPU and requires
the matching CUDA 13 dependency lock. From the matching source runtime:

```bash
RTP_TEST_B12X_NVFP4_GPU=1 python -m \
  rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.nvfp4_hybrid_executor_test \
  B12xNvfp4GpuTest
```

The Qwen4 FP8-to-NVFP4 recipe selects Hybrid automatically for compatible
SM120 TP2/EP2 Decode workers. The static backend requires this recipe.
`decode_b12x_max_rows` accepts 0 through 16 physical input rows.
Set it to 0 to use CUTLASS for all rows. Set `RTP_NVFP4_HYBRID_MOE=0` to
pin the existing CUTLASS executor for comparisons. Omit
`decode_cutlass_profile_ids` to use the runtime's normal tactic selection.
Explicit tactic IDs are specific to the FlashInfer build and GPU shape, so
validate them on the deployment wheel.

The CUDA 13 native extension includes the SM120 TP2 shared-expert LSA
fusion. Both Decode ranks check the packaged NCCL API, model layout,
CUDA peer access and free memory before constructing the communicator.
An explicit NCCL P2P or cuMem disable, or any other failed preflight,
selects the standard routed collective on both ranks. Once both ranks
enter communicator creation, a native failure aborts startup for the TP
group. A unilateral fallback at that point could split the collective.
Both ranks use the same logical batch shape to select the epilogue and size
its grid. The grid has at most 64 CTAs. Each CTA uses its block index as
one of the 64 reserved barrier slots, and the final barrier protects the
symmetric input window until both peers have finished reading it.
The acquire/release barriers publish the staged partial before peer reads
and retire peer reads before the next layer reuses the window. A missing
peer triggers a CUDA launch failure at the timed barrier (nominally 10
seconds). A rank-local failure after route reduction has been suppressed
requires a restart of the TP group.
[NCCL local AllReduce using LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#local-allreduce-using-lsa)
[NCCL 2.30.4 timed LSA barrier](https://github.com/NVIDIA/nccl/blob/v2.30.4-1/src/include/nccl_device/impl/lsa_barrier__funcs.h#L168)

Run `//rtp_llm/models_py/modules/qwen4_exp/test:reduce_gate_integration_test`
on CUDA 13. The test reserves two local SM120 GPUs, and it is included in
`qwen4_moe`. The source-runtime invocation is:

```bash
NCCL_P2P_DISABLE=0 python -m \
  rtp_llm.models_py.modules.qwen4_exp.test.moe.reduce_gate_integration_test
```

The manual `reduce_gate_missing_peer_test` target makes rank 1 omit the
collective and checks the resulting CUDA fault on rank 0.

On the admitted BF16 Decode path, the shared projections and gate run on
an auxiliary CUDA stream and join the routed result before the LSA epilogue.
For a static Qwen4 checkpoint, set `"decode_shared_gemm_nk": true` in the
ModelOpt FP4 quantization config to also pack contiguous BF16 shared
projection weights. The measured shapes use packed weights for up at
physical M1–16 and down at M2–32. A 48-layer Qwen3.8-Flash-Next worker
uses an additional 450 MiB per rank. This option defaults to false.
Updating packed weights requires restarting the model and rebuilding the
CUDA Graphs.

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
