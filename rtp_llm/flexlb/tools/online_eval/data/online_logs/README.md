# 历史性能 UT 兼容夹具

本目录只供 `flexlb-api` 的 `MasterBatchEndToEndPerformanceTest` 使用，保留历史测试的输入契约。它不是现行 online_eval 的流量数据源，也不是引擎性能模型。

- `sample_access.json` 提供脱敏 token 素材、模型和生成参数。
- `trace_30min.jsonl` 提供历史请求形状；解释、筛选和模板构造以该 UT 为准。
- AI 与开发者不得将本目录用于新 case、默认流量、性能标定或真实流量画像；不得自动迁入 `traffic_trace/` 或 `performance/`。
- 不因公共数据治理、匿名化或格式统一而删除、重生成、压缩替换这些文件。修改夹具或 UT 输入规则必须获得明确授权，并单独验证测试负载是否改变。
- 本目录不纳入流量模型自动发现；现行框架数据选择遵循上级 [数据说明](../README.md)。

兼容来源是 main 提交 `e73826e27a6005640bb75ce2a73f235bfa51c55b`，两份夹具保持原始字节。此提交标识固定输入来源，不代表当前实验运行版本。
