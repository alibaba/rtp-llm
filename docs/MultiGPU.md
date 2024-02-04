# 多卡并行
rtp-llm支持单机多卡并行

## 单机多卡
单机多卡并行配置非常简单，只需在启动服务时添加环境变量`TP_SIZE`, `WORLD_SIZE`，请求服务则和单卡时的逻辑一致，参考命令如下: 
``` python
TP_SIZE=2 WORLD_SIZE=2 python3 -m maga_transformer.start_server
```