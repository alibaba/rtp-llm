# Multi-GPU Parallelism
rtp-llm supports single-node multi-GPU, multi-node single-GPU, and multi-node multi-GPU parallel strategies.
Multi-node parallelism requires rtp-llm version >= `0.1.11`.

## Single-Node Multi-GPU
To use single-node multi-GPU parallelism, you need to add environment variables `TP_SIZE` and `WORLD_SIZE` when starting the service. The request logic is consistent with single-GPU, refer to the following command:
``` python
TP_SIZE=2 WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m rtp_llm.start_server
```

## Multi-Node Single/Multi-GPU
When starting the service, you need to configure the environment variable `DISTRIBUTE_CONFIG_FILE=/path/to/file`. The configuration content is JSON with the following format. The port is optional; if not filled, it is considered the same as the master port:
``` json
{
    "llama13B_2A10_PCIE_1_inference_part0": {
        "name": "llama13B_2A10_PCIE_1_inference_part0",
        "ip": "11.161.48.116",
        "port": 10000
    },
    "llama13B_2A10_PCIE_1_inference_part1": {
        "name": "llama13B_2A10_PCIE_1_inference_part1",
        "ip": "11.161.48.116",
        "port": 20000
    }
}

```
The key in JSON and the name in the value should be consistent. The service will establish collective communication with the machine suffixed with `_part0` as rank0. At the same time, you need to configure `WORLD_SIZE`, `TP_SIZE`, and `TP_RANK`.

### Multi-Node Single-GPU Startup Command
rank0:
``` shell
START_PORT=10000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=0 TP_SIZE=2 WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m rtp_llm.start_server
```
rank1:
``` shell
START_PORT=20000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=1 TP_SIZE=2 WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m rtp_llm.start_server
```

### Multi-Node Multi-GPU Startup Command
When `LOCAL_WORLD_SIZE` > 1, `WORLD_SIZE` % `LOCAL_WORLD_SIZE` == 0 is required. At this time, `LOCAL_WORLD_SIZE` GPUs will be used for inference on each machine. When setting `WORLD_RANK` for machines, it needs to be multiplied by `LOCAL_WORLD_SIZE`.

rank0
``` shell
START_PORT=10000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=0 TP_SIZE=4 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m rtp_llm.start_server
```
rank1:
``` shell
START_PORT=20000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=2 TP_SIZE=4 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m rtp_llm.start_server
```

### Accessing the Service
The service endpoint is the Uvicorn Server address of rank0, i.e., http://0.0.0.0:10000

## Common Issues and Solutions
1. Startup error `Caught signal 7 (Bus error: nonexistent physical address)`
Single-node multi-GPU communication uses shared memory, and this error is due to insufficient shared memory.
- If started via Docker, add the `--shm-size=2g` parameter
- If deployed via K8s, shared memory can be set by adding a Memory-type volume to the container.
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: your-model-pod
spec:
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: "2Gi"
  containers:
    - name: mycontainer
      image: your_image_name
      volumeMounts:
        - name: dshm
          mountPath: /dev/shm
```