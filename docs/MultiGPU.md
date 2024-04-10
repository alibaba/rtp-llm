# 多卡并行
rtp-llm支持单机多卡并行

## 单机多卡
使用单机多卡并行，需要在启动服务时添加环境变量`TP_SIZE`, `WORLD_SIZE`，请求服务时和单卡逻辑一致，参考命令如下: 
``` python
TP_SIZE=2 WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```

## 多机单/多卡
在起服务时需要配置环境变量`DISTRIBUTE_CONFIG_FILE=/path/to/file`，配置内容为json，格式如下:
``` json
{
    "llama13B_2A10_PCIE_1_inference_part0": {
        "name": "llama13B_2A10_PCIE_1_inference_part0",
        "ip": "33.76.194.173"
    },
    "llama13B_2A10_PCIE_1_inference_part1": {
        "name": "llama13B_2A10_PCIE_1_inference_part1",
        "ip": "33.76.194.182"
    }
}

```
json的key和value中name保持一致，服务会以后缀为`_part0`的机器作为rank0建立集合通信。同时需要配置`WORLD_SIZE`, `TP_SIZE`, `TP_RANK`
### 多机单卡启动命令
rank0:
``` shell
DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=0 TP_SIZE=2 WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
rank1:
``` shell
DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=1 TP_SIZE=2 WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
### 多机多卡启动命令
当`LOCAL_WORLD_SIZE` > 1时，需要`WORLD_SIZE` % `LOCAL_WORLD_SIZE` == 0, 此时会在每台机器使用`LOCAL_WORLD_SIZE`张卡进行推理，此时机器设置`WORLD_RANK`需要乘以`LOCAL_WORLD_SIZE`

rank0
``` shell
DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=0 TP_SIZE=4 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
rank1:
``` shell
DISTRIBUTE_CONFIG_FILE=/path/to/file WORLD_RANK=2 TP_SIZE=4 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MODEL_TYPE=xxx CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer python3 -m maga_transformer.start_server
```
