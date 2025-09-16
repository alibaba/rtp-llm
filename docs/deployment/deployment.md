# Deployment Guide Using Kubernetes

This guide walk you through deploying the RTP-LLM service on Kubernetes. You can deploy RTP-LLM to Kubernetes using any of the following:

- [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [StatefulSet](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [LWS](https://lws.sigs.k8s.io/docs/overview/)

## Deploy with Kubernetes Deployment
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

## Multi-Node Deployment
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
