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


```bash
# Use the last release branch
git clone -b release/0.0.1 git@gitlab.alibaba-inc.com:foundation_models/RTP-LLM.git
cd RTP-LLM

# build RTP-LLM whl target
# --config=cuda12_6 build target for NVIDIA GPU with cuda12_6
# --config=rocm build target for AMD
bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=cuda12_6 --test_output=errors --test_env="LOG_LEVEL=INFO"  --jobs=64

ln  -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/proto/model_rpc_service_pb2.py  `pwd`/rtp_llm/cpp/proto/

```


## Method 3: Using docker
More Docker versions can be obtained from [RTP-LLM Release](../release/index.rst)
```bash
docker run --gpus all \
 --shm-size 32g \
 -p 30000:30000 \
 -v /mnt:/mnt \
 -v /home:/home \
 --ipc=host \
 hub.docker.alibaba-inc.com/isearch/rtp_llm_gpu_cuda12:2025_07_08_21_00_a1ed8e8 \
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
      image: hub.docker.alibaba-inc.com/isearch/rtp_llm_gpu_cuda12:2025_07_08_21_00_a1ed8e8
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
          image: hub.docker.alibaba-inc.com/isearch/rtp_llm_gpu_cuda12:2025_07_08_21_00_a1ed8e8
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
          image: hub.docker.alibaba-inc.com/isearch/rtp_llm_gpu_cuda12:2025_07_08_21_00_a1ed8e8
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
