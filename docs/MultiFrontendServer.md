# 背景
* 为了扩展python层的性能，避开GIL锁的问题。
* 主要是为了render和tokenizer的性能问题等。
* 将推理的前端进程拆分开，并且可以配置为多进程模式，从而提升python层的性能。

## 使用方法
* 配置环境变量FRONTEND_SERVER_COUNT=x，x为整数，代表前端进程的个数。
* start server的使用方法和之前一样。
* 执行ps xf，看到前端和后端进程的名字和个数。
* 也可以手动单独启动前端进程。
* 使用和start server相同的env，然后执行 /opt/conda310/bin/python3 -m maga_transformer.start_frontend_server

## 注意：
* 在openai，raw 两种请求格式下，会有优势，如果这里的python部分出现了性能瓶颈，可以配置成多进程前端模式。
* 在其他场景没有收益。