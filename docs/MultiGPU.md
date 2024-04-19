# 多卡并行
rtp-llm支持单机多卡、多机单卡、多机多卡并行策略。  
多机并行需要rtp-llm版本>=`0.1.11` 。

## 单机多卡
使用单机多卡并行，需要在启动服务时添加环境变量`TP_SIZE`, `WORLD_SIZE`，请求服务时和单卡逻辑一致，参考命令如下: 
``` python
TP_SIZE=2 WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```

## 多机单/多卡
在起服务时需要配置环境变量`DISTRIBUTE_CONFIG_FILE=/path/to/file`，配置内容为json，格式如下，其中port为可选项，如果不填则认为和master端口相同:
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
json的key和value中name保持一致，服务会以后缀为`_part0`的机器作为rank0建立集合通信。同时需要配置`WORLD_SIZE`, `TP_SIZE`, `TP_RANK`
### 多机单卡启动命令
rank0:
``` shell
START_PORT=10000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=0 TP_SIZE=2 WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
rank1:
``` shell
START_PORT=20000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=1 TP_SIZE=2 WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
### 多机多卡启动命令
当`LOCAL_WORLD_SIZE` > 1时，需要`WORLD_SIZE` % `LOCAL_WORLD_SIZE` == 0, 此时会在每台机器使用`LOCAL_WORLD_SIZE`张卡进行推理，此时机器设置`WORLD_RANK`需要乘以`LOCAL_WORLD_SIZE`

rank0
``` shell
START_PORT=10000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=0 TP_SIZE=4 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
rank1:
``` shell
START_PORT=20000 DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=2 TP_SIZE=4 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
### 访问服务
服务端点为rank0的Uvicorn Server地址，即http://0.0.0.0:10000

## 常见问题及解决方案
1. 启动报错`Caught signal 7 (Bus error: nonexistent physical address)`  
单节点多卡间采用共享内存的方式通信，该报错是因为共享内存不足。  
- 如通过docker启动，需添加`--shm-size=2g`参数
- 如通过K8s部署，可通过为容器添加Memory类型的存储卷设置共享内存。
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