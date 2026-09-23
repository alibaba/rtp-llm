# 测试数据

数据分为真实流量、合成流量画像和引擎性能参数。实验输出写入 `run/` 或指定归档目录。
新增数据提交到仓库前必须取得用户同意；不需要登记 catalog、修改白名单或为数据新增 case。
已有数据的重命名不改变内容。不可重采的原始样本应保存于仓库外归档。

| 目录 | 内容 |
|---|---|
| `traffic_models/` | 真实流量的匿名 prefix DAG 压缩文件及同名 manifest；保留请求顺序、到达间隔和 prefix 结构，不含真实文本或 token。`.templates.json` 是从真实流量截取的 Java 回归夹具。 |
| `calibration/` | 参数化合成流量的统计画像。即使画像来自真实数据，生成的请求也是合成流量，不是原请求回放。 |
| `performance/` | 引擎计算耗时模型，与流量来源分类独立；预设名称由 `config/performance_presets.json` 解析。 |

## 选择数据

`run_stress.py --traffic-model <文件名去掉.xz>` 直接读取 `traffic_models/`，`--help` 列出可选文件。
无需中央清单。参数化合成通过 `--traffic-source-spec` 选择 `synthetic/realistic/1` 源；
其 `profile` 按 `calibration/` 下去掉 `.profile.json` 的文件名选择，如
`glm-5.3_20260921_1400_15m`；也可直接填写生成参数，无需增加登记项。
现有场景继续在 YAML 中固定输入文件及 SHA，不因目录增加数据而改变默认输入。

| 真实流量文件名 | 采集窗口（北京时间） | 请求数 | 观测平均 QPS |
|---|---|---:|---:|
| `glm-5.3_20260921_1400_15m.xz` | 09-21 14:00 至 14:15 | 141,113 | 156.8 |
| `glm-5.3_20260921_2126_3h.xz` | 09-21 21:26 至 09-22 00:25 | 587,711 | 54.7 |
| `glm-5.3_20260922_0610_3h30m.xz` | 09-22 06:10 至 09:40 | 538,208 | 42.7 |

时长名称经过取整，精确窗口在 manifest。平均 QPS 使用捕获事件数 / 首末事件跨度，
不对缺失 Pod 外推。压测发送节奏由客户端参数控制。默认仍为首份 15 分钟数据；
另两份尚未标定门禁，切换数据不能直接沿用旧基线结论。

文件名优先使用已确认的业务来源或模型，再加采集起点和时长；本批服务模型由用户确认为 `glm-5.3-model`，文件名使用 `glm-5.3`。
Spectrum 业务来源标识仍待补充，`frontend` 仅表示采集位置。codec 版本与 SHA 放在 manifest，不用作统一文件名前缀。
`calibration/glm-5.3_20260921_1400_15m.profile.json` 是首份捕获派生的合成画像。
对应 `.templates.json` 可运行 `python3 scripts/pipeline/derive_master_templates.py` 重建。

## Manifest

每份真实流量旁有同名 `.manifest.json`：

- `source.capture`：采集层和已知部署，不能替代业务来源或模型。
- `source.spectrum`：Spectrum 业务来源标识、确认状态、信息依据。目前没有自动化查询链，需人工补充。
- `source.model`：真实服务模型名称、确认状态、信息依据。未知时 `name: null, status: unconfirmed`，不从部署名或 mock 性能配置推断。
- `capture_window`、`provenance`：可读时间窗，以及文件内保留的 Pod 覆盖和原始来源校验信息。
- `statistics`：请求数、平均 QPS、含空闲桶的每秒请求数分位数、输入长度分布及总量（v3 精确长度，v2 块对齐）、prefix 共享分布。
- `codec`、`bytes`、`sha256`：解码格式和文件完整性；原始请求文本不入库。

统计范围仅是捕获请求。分位数采用 nearest-rank；每秒桶从首个事件开始，包含最后一个可能不足一秒的桶。
共享比例是 prefix DAG 的理论结构共享，不能当真实缓存命中率。当前编码未保留输出长度和成功/失败状态，
这些统计明确为 `null`。合成画像中的历史拟合统计不替代当前真实流量文件的统计。

重算统计会保留已手工补充的 `source`：

```bash
python3 scripts/pipeline/describe_traffic.py data/traffic_models/glm-5.3_20260921_1400_15m.xz
```

也可通过 `--source-info /path/to/source.json` 提供完整 `source` 对象。
确认后填写 Spectrum `identity`、模型 `name`、`status: confirmed` 及各自的 `evidence`；
未确认字段保持 `null/unconfirmed`。这一步描述已存在的文件，不自动授权提交新数据。
`fit_frontend_prefix.py` 拟合时同时写出这份 sidecar。

## 性能参数

性能文件采用 `<model>_<hardware-or-calibration>.json` 命名。
`synthetic_baseline.json` 与 `synthetic_prefill_100ms.json` 是合成测试性能模型；
`deepseek_v4_flash_decode_table.json` 保留 decode 表标定，`deepseek_v4_flash_sm100.json` 是 SM100 开发环境参数；
`deepseek_v4_flash_l20c.json` 来自 Flash 模型 L20C 测试部署的标定（prefill 公式 ×1.23、decode step 模型、EOS 均值 400）；
`glm_5_3_l20d.json` 来自 GLM-5.3 L20D 对齐实验。它们的模型名不代表上述流量的服务模型。

合成画像 schema 2 保存联合分布；默认仍保持独立采样的旧 seed 语义。显式 `sampling: joint` 使用联合采样，
独立对比工具及限制见[合成保真度](../docs/reference/concepts/synthetic-fidelity.md)。

## 采集契约与归档

[capture_contract.py](../src/traffic/capture_contract.py) 是 capture→fit 行契约的唯一出处，
声明字段名、类型、可空性、语义及匿名化规格；capture/fit 均执行它的校验。
行文件和 summary 都记录 `schema_version` 与 `block_size`。块大小在该模块单点声明，
默认 512；修改后采集、拟合和 codec 同步使用新值，不匹配的历史件会被拒绝，不能混用。
缺到达时间戳等不可用日志行在 summary 显式计数；契约字段不能静默补齐。
无 schema 的旧行文件不被自动猜测接纳，需根据真实来源核验契约，不能用拟合件冒充。

仓库外按 `<来源>/<UTC起止窗口>/capture/pod-N.jsonl.gz`（或 `.jsonl.xz`）及
`capture/pod-N.summary.json` 成对归档；Pod 索引映射和来源信息随窗口保存。
这是“拟合前原始中间件”的唯一来源，不含原始 token。派生文件保存到同窗口的
`fit/v3/`（历史复拟合为 `fit/v2/`），包括 `.xz`、manifest 和 fit-report。
行文件的压缩字节 SHA256 由 summary 绑定，fit-report 保存所消费的 summary。
**以拟合后文件冒名 original 属事故**：`.xz` 扩展名不能证明它是原始中间件。
不覆盖原始归档，不以拟合件恢复、推算或生成已丢失的精确长度。

采集支持 `--log-dir`、`--log-glob`；运行位置、部署/Pod 定位由外部执行环境负责。
护栏默认 `--time-budget-s 900 --tail-bytes 16000000 --completion-grace-ms 300000`。
超预算默认 `--on-budget error` 返回失败，仍写 summary；显式 `truncate` 正常返回，
但 summary 必须标记 `complete: false`、`truncated: true` 和 `budget_exceeded`。
fit 拒绝不完整窗口。增加预算或分成更小的到达窗口重新采集，再分别拟合；
不实现基于文件偏移的续采，以免轮转后的偏移被误认作同一日志。

## 编码代际与复算

新采集默认拟合 v3：保留精确输入总长，仅完整块参与前缀匹配，残缺尾块私有。
显式 `--model-version 2 --v2-reason '历史复拟合理由'` 才生成 v2，理由进入 provenance。
v2 读取路径保留。现有 v2 件、SHA、画像、夹具与场景 pin 是历史工件，不能原地换代。
唯一退役路径是“重采新窗口 → v3 文件与 manifest 入库 → 引用方换代重标定”。
文件发现、manifest 校验、`run_stress --traffic-model` 与场景 SHA pin 同时支持 v2/v3；
不因 v3 成为默认而修改既有默认门禁数据或阈值。

禁止以格式转换或尾长生成冒充实测 v3。若另行开展尾长合成实验，manifest 必须
独立声明合成来源，产物留在实验归档，不得进入真实流量目录或门禁输入。
同场景跨 codec 代际或 tail 语义不能直接比较命中率/TPS 绝对值、32k 过滤后的请求集；
现有 `comparison_notice` 通道提示该差异，重标定后才能建立新的对照结论。
