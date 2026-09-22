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
- `statistics`：请求数、平均 QPS、含空闲桶的每秒请求数分位数、块对齐输入长度分布及总量、prefix 共享分布。
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
编译 v1 模型时，`python3 -m traffic.prefix_lineage` 会同时写出这份 sidecar。

## 性能参数

性能文件采用 `<model>_<hardware-or-calibration>.json` 命名。
`synthetic_baseline.json` 与 `synthetic_prefill_100ms.json` 是合成测试性能模型；
`deepseek_v4_flash_decode_table.json` 保留 decode 表标定，`deepseek_v4_flash_sm100.json` 是 SM100 开发环境参数；
`glm_5_3_l20d.json` 来自 GLM-5.3 L20D 对齐实验。它们的模型名不代表上述流量的服务模型。

合成画像 schema 2 保存联合分布；默认仍保持独立采样的旧 seed 语义。显式 `sampling: joint` 使用联合采样，
独立对比工具及限制见[合成保真度](../docs/reference/concepts/synthetic-fidelity.md)。
