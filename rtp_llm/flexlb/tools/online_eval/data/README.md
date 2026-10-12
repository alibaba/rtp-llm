# 测试数据

本页只说明数据规则和用法，遵循 [文档维护规则](../AGENTS.md)。

数据分为真实流量、合成流量画像和引擎性能参数。实验输出写入 `run/` 或指定归档目录。
新增数据提交到仓库前必须取得用户同意；通过文件发现选择数据，无需增加 case。
不可重采的原始样本应保存于仓库外归档。

| 目录 | 内容 |
|---|---|
| `traffic_trace/` | 真实流量的匿名 prefix DAG 压缩文件及同名 manifest；保留请求顺序、到达间隔和 prefix 结构，不含真实文本或 token。Java 回归夹具按需从固定快照派生；拟合诊断和完成状态留在未入库的 `run/` 或仓库外实验归档。 |
| `synthetic_parameters/` | 参数化合成流量的统计画像。即使画像来自真实数据，生成的请求也是合成流量，不是原请求回放。 |
| `performance/` | 引擎计算耗时模型，与流量来源分类独立；预设名称由 `config/performance_presets.json` 解析。 |

## 选择数据

case 在 YAML 的流量源声明中选择 `traffic_trace/` 下的文件。参数化合成使用 `synthetic/realistic/1` 源；
其 `profile` 按 `synthetic_parameters/` 下去掉 `.profile.json` 的文件名选择，也可显式填写生成参数。
场景在 YAML 中固定输入文件及 SHA，不因目录增加文件而改变默认输入。
来源、窗口、请求数、统计值、SHA 和反推参数从对应 manifest/profile/config 读取，不在本页列清单。

## 命名与来源

流量文件优先使用已确认的业务来源或模型，加采集起点和时长。精确窗口以 manifest 为准；
codec 版本与 SHA 放在元数据中，不作为统一文件名前缀。`frontend` 只表示采集位置，
不能替代业务身份。更换来源、模型或长度口径后重新确认可比性，不直接沿用其他输入的门禁结论。
性能文件采用 `<model>_<hardware-or-capture>.json`，性能模型与流量来源分别标识。

## Manifest

每份真实流量旁有同名 `.manifest.json`。新输出使用 `traffic_manifest_schema_version: 1`，顶层固定为：

- `artifact`：压缩字节数与 SHA、codec 名称与版本、编码元数据（block、长度、tail、arrival 语义）。codec 版本只在一处声明，读取时与数据 envelope 核对。
- `capture`：来源归属、申请窗口 `requested_window`、保留样本首末时间 `observed_window`、分片覆盖与限制。`execution` 独立保存执行时间、工具、问题和原始归档引用；开始/结束不能从请求首末时间推断，未知历史事实标 `unavailable`。
- `transformation`：拟合工具内容 SHA、原始分片 SHA 与 summary。只能在原始证据存在时标 `recorded`；旧数据重算统计不能升级为已核验采集。
- `statistics`：选择数据所用的分布与统计口径，不参与补齐编码信息。block 统计读取当前数据的 block 大小。

历史 `manifest_schema_version: 2` sidecar 保持字节不变；读取边界校验后投影为统一运行字段。这里的版本表示两种不同的存储格式，不是回放 codec 代际。拒绝未知版本及文件/sidecar 不一致，不猜测缺失字段。新拟合和统计描述入口写分层格式，重算同一 SHA 时保留来源、人工限制与已有转换证据。

统计范围仅是捕获请求。分位数采用 nearest-rank；每秒桶从首个事件开始，包含最后一个可能不足一秒的桶。
共享比例是 prefix DAG 的理论结构共享，不能当真实缓存命中率。当前编码未保留输出长度和成功/失败状态，
这些统计明确为 `null`。合成画像中的历史拟合统计不替代当前真实流量文件的统计。

重算统计会保留已手工补充的 `source`：

```bash
python3 scripts/pipeline/describe_traffic.py /path/to/model.xz
```

也可通过 `--source-info /path/to/source.json` 提供完整 `source` 对象，通过 `--capture-info /path/to/capture-execution.json` 提供独立采集执行事实。
确认后填写来源 `identity`、模型 `name`、`status: confirmed` 及各自的 `evidence`；
未确认字段保持 `null/unconfirmed`。这一步描述已存在的文件，不自动授权提交新数据。
`fit_frontend_prefix.py` 拟合时同时写出这份 sidecar。

## 派生数据

`python3 scripts/pipeline/derive_master_templates.py --model /path/to/model.xz --out /path/to/run/model.templates.json`
在运行目录生成 Java 回归夹具。合成画像由 `traffic.synthetic.derivation` 从固定模型与 fit-report 反推参数；
它描述统计分布，不等同于原请求回放。`sampling: joint` 的用法和限制见
[合成保真度](../docs/development/synthetic-fidelity.md)。运行命令的工作目录为 `online_eval`，
模块入口需要 `PYTHONPATH=src:.`。

## 采集契约与归档

[capture_contract.py](../src/traffic/capture_contract.py) 是 capture→fit 行契约的唯一出处，
声明字段名、类型、可空性、语义及匿名化规格；capture/fit 均执行它的校验。
行文件和 summary 都记录 `schema_version` 与 `block_size`。块大小在该模块单点声明，
默认 512；修改后采集、拟合和 codec 同步使用新值，不匹配的历史件会被拒绝，不能混用。
缺到达时间戳等不可用日志行在 summary 显式计数；契约字段不能静默补齐。
无 schema 的旧行文件不被自动猜测接纳，需根据真实来源核验契约，不能用拟合件冒充。

仓库外按 `<来源>/<UTC起止窗口>/capture/shard-N.jsonl.gz`（或 `.jsonl.xz`）及
`capture/shard-N.summary.json` 成对归档；分片身份映射和来源信息随窗口保存。
这是“拟合前原始中间件”的唯一来源，不含原始 token。派生文件保存到同窗口的
`fit/v3/`（历史复拟合为 `fit/v2/`），包括 `.xz`、manifest 和 fit-report。
行文件的压缩字节 SHA256 由 summary 绑定，fit-report 保存所消费的 summary。
**以拟合后文件冒名 original 属事故**：`.xz` 扩展名不能证明它是原始中间件。
不覆盖原始归档，不以拟合件恢复、推算或生成已丢失的精确长度。

采集支持 `--log-dir`、`--log-glob`；运行位置、部署/Pod 定位由外部执行环境负责。
护栏默认 `--time-budget-s 900 --tail-bytes 16000000`。采集窗口按请求到达时间筛选，完成时间不用于裁剪窗口。
超预算默认 `--on-budget error` 返回失败，仍写 summary；显式 `truncate` 正常返回，
但 summary 必须标记 `complete: false`、`truncated: true` 和 `budget_exceeded`。
fit 拒绝不完整窗口。增加预算或分成更小的到达窗口重新采集，再分别拟合；
不实现基于文件偏移的续采，以免轮转后的偏移被误认作同一日志。

## 性能采集包

原始配置、测量证据和 Mock 参数分开保存。现有 `performance/` 参数文件保持不变，继续由 preset 加载；缺少历史来源的文件保持 `legacy_unverified`，不得将当前 Whale 配置回填为历史现场。新采集包在仓库外构建，通过 `performance_capture_schema_version: 1` 的 manifest 关联已有文件：

- `identity`：采集包 ID、模型、硬件、不透明 `deployment_ref`。真实 ID 映射和未脱敏响应放受控归档；入仓副本须先完成脱敏。SHA 不承担加密作用。
- `collection`：带时区的采集执行时间、工具版本、`complete/partial/legacy_unverified` 状态及问题清单。完整性不等于拟合准确性。
- `configuration`：`template/biz/deployment/carbon_target/process_readback` 各层分别声明可用性、证据文件与 SHA；各层明确四个 role 的可用性、镜像、可得的 commit 与版本依据。不存在的层或角色用 `unavailable/not_applicable` 和理由表示，不复制其他层补齐。
- `measurements`：Prefill/Decode/EOS 等测量引用证据文件、实际采用的配置层、方法及单位。新增测量种类可追加条目，不改加载器；EOS 实测分布和模拟的 geometric/seed 分属测量与派生层。
- `derived`：拟合结果或 Mock 参数文件，绑定测量 ID 与推导方法。运行 preset 只读取显式选中的 Mock 参数，不自动读取整包或改变参数。

Whale MCP 只读顺序为：部署详情与 `deployment_images` → 部署实际 `biz_version` 的快照 → 对应 template version → 各 zone 的 Carbon target。服务最新版本不能代替部署版本；镜像卡型候选不能自动归属到角色。Carbon target 是调度目标，不是进程实际配置；只有真实进程回读才能标 `process_readback`。

`plan_whale_capture.py` 从 MCP 的部署 data 与镜像返回对象产生固定版本 GET 计划，不执行网络请求、不推断角色 commit。平台读取与脱敏由调用方负责；本地接口校验引用文件、SHA、角色来源和测量依赖，不自动采集、不存储在线凭据、不加密身份。

```bash
python3 scripts/pipeline/plan_whale_capture.py --deployment /path/to/deployment-data.json --images /path/to/deployed-images.json
python3 scripts/pipeline/validate_performance_capture.py /path/to/bundle/manifest.json
python3 scripts/pipeline/validate_performance_capture.py /path/to/record-spec.json --publish /path/to/bundle/manifest.json
```

发布前所有证据文件必须已位于 bundle 内，拒绝路径逃逸、SHA 不符、伪造运行时版本依据与覆盖已有 manifest。只校验已有数据不会授权采集或入库新样本。

## 模型采集档案与口径

模型侧真实观测值只在一个 `data/performance/` 采集档案定义。档案须携带部署身份、采集窗口和完整性字段；缺原始记录的现有档标为 `legacy_unverified`，不能宣称已核验。只含局部参数且无消费者的档标为 `orphan`，不得被新 case 当作权威采集。变更性能/容量先改档案；场景中的测试偏离必须声明基线和理由。

性能档案可用 `master.config_overrides` 捆绑配套 Master 参数，使用与场景
`environment.config_overrides` 相同的字段；`master.provenance` 记录 `status` 和 `source`。
记录校验和覆盖 Engine 与 Master 整套内容，修改任一侧须更新校验和。
加载器分别投影 Mock 性能与 Master 配置，不能把 Master 字段传给 Mock Engine。
场景选中预设即继承配套参数；显式覆盖配套值须在 `model_override` 声明基线和原因。
Master 运行形态仍由所选 profile 明确指定。未核验的历史实验配置保持
`legacy_unverified`，不因为与 Engine 数据放在一起而成为已核验的生产采集。

`prefill_kv_pool_blocks`/`decode_kv_pool_blocks` 是 Java mock 的 KV 池块数；`prefill.memory_cache.capacity_blocks` 是内存前缀树容量；性能 JSON 顶层 `block_size` 是引擎时间模型块口径；capture 行契约的 `BLOCK_SIZE` 是流量前缀摘要块口径。这四者不得以裸称“blocks”混用，也不要求数值相等。mock 的 prefill/decode block-size 是档案里的独立运行参数，读取方必须按各自口径传入。

真实快照的内容变换、记录字段和播放边界以 [流量架构契约](../docs/architecture/traffic.md) 为准。合成画像保留历史格式键 `calibration` 与 `held_out_validated`，目录名不改变旧工件格式或 SHA。

DSv4 mock 测试刻度由 `data/performance/dsv4_l20_mock_calibration.json` 保存，`default` 预设直接读取该文件；其状态为 `legacy_unverified`，不是已验证的线上测量。预设注册表通过相对路径选择性能文件或引用刻度，加载时将 decode 数值、模型身份及刻度 SHA 写入运行用性能 JSON；master 默认 prefill 表达式读取同一文件。Java mock jar 将该文件作为资源打包，供无显式 FORMULA 的独立启动使用；LEARNING 在 mock 中使用这份静态近似。采集档案保持自己的记录、哈希与 block 口径，不与该测试刻度共用身份。

## 编码代际与复算

新采集默认拟合 v3：保留精确输入总长，仅完整块参与前缀匹配，残缺尾块私有。
显式 `--model-version 2 --v2-reason '历史复拟合理由'` 才生成 v2，理由进入 provenance。
v2 读取路径保留。现有 v2 件、SHA、画像、夹具与场景 pin 是历史工件，不能原地换代。
唯一退役路径是“重采新窗口 → v3 文件与 manifest 入库 → 引用方换代重标定”。
文件发现、manifest 校验、流量物化与场景 SHA pin 同时支持 v2/v3；
不因 v3 成为默认而修改既有默认门禁数据或阈值。

禁止以格式转换或尾长生成冒充实测 v3。若另行开展尾长合成实验，manifest 必须
独立声明合成来源，产物留在实验归档，不得进入真实流量目录或门禁输入。
同场景跨 codec 代际或 tail 语义不能直接比较命中率/TPS 绝对值、32k 过滤后的请求集；
对照前应核对双方 `flow-input.json` 的来源版本、tail 与播放语义，重标定后才能建立新的对照结论。
