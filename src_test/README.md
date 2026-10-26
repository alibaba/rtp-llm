# RTP-LLM sCR 复现材料

这里保存与工作区源码对应的 sCR/Epsilon、PD-fusion 和 CUDA Graph 复现实验脚本及结果文档。脚本默认使用本机已经安装的 `/opt/conda310` RTP-LLM 包、`/etc/scr/epsilon` shim 和 `/home/yuziqu.yzq/scr_controller`；模型路径、端口和 checkpoint 路径都可以通过环境变量覆盖。

## 目录

- [`decode_pd_graph_20260904/`](decode_pd_graph_20260904/)：DECODE CUDA Graph + PREFILL PD-fusion 启动、请求、GPU-only dump/restore 流程。
- [`expandable_segments_ab_20260904/`](expandable_segments_ab_20260904/)：`expandable_segments=True/False` 对照启动脚本和结果。

## 当前保存的结果

- PD 路由请求已经得到 HTTP 200，`aux_info.pd_sep=true`，证明请求从 gateway 进入 PREFILL 并远程调用 DECODE。
- DECODE 使用 `--enable_cuda_graph 1 --decode_capture_config 1,2,4,8,16`；PREFILL 保持 CUDA Graph 关闭。
- 本轮 GPU dump 的 controller RPC 返回 0，4 个 GPU 文件已写入；但 dump 后主机发生全局 OOM，scheduler/ttrpc 失联，restore 在用户要求保存进度时中止，不能把本轮称为完整 restore roundtrip。

执行前请先确认 `/tmp/test` 和 `/dev/shm` 有足够空间，并保留 scheduler 配置文件，不要为了实验修改 `/home/yuziqu.yzq/scheduler_config.json`。
