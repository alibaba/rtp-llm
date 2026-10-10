# 合成输入保真度与独立观察工具

参数化合成用于控制变量实验。它保留家族与 Zipf 抽象，不能仅凭输入长度相似就当作真实 cache 工作负载。
使用 `scripts/commands/compare_traffic.py` 独立对比生产生成器和真实捕获；它输出 HTML、Markdown、JSON，
不会接入实验报告层，也不会在普通压测或物化时自动生成报告。

## 采用的方法

生产 `realistic` 现有两种显式采样语义：

- `sampling: independent`：默认兼容路径，长度经验 ICDF 加固定家族深度。保留旧 seed 的请求字节；只有这一份兼容实现，报告侧不再复制独立采样生成器。
- `sampling: joint`：从暖请求 `(blocks, shared_blocks)` 联合池采样，并从冷请求长度池采样冷请求。每个家族维护可增长的固定前缀，取联合对指定的深度，再分配私有后缀。允许真实数据中的全长共享。家族首次出现或首次增长的前缀仍是冷的，分析时照实计入。

联合池由真实捕获按 `(blocks, shared_blocks)` 排序，每个等质量层取中点代表，最多 1024 对；
冷请求长度池同样最多 1024 点。池保存在 schema 2 profile 的 `parameters.joint_distribution`，
不保存原始文本、token 或事件顺序。schema 1 画像仍支持独立采样；没有联合字段时显式请求 joint 会失败，
不会静默推断新结构。schema 2 也不会自动切换现有 seed 的采样方法。

两种方法共用生产请求迭代器和写盘实现。长度 CDF 预计算一次，避免每条请求重复构造权重累计数组；
独立采样的 RNG 次序不变。输出长度仍用独立 RNG、由用户指定。joint 不支持 pinned/global prefix/session 生长覆盖，
也不接受独立长度/深度覆盖，避免混合含义不清的规则。家族数、热度与冷比例仍可作受控实验变量。

合成 source 示例：

```json
{"kind":"synthetic","model":"realistic","version":"1","parameters":{"profile":"<profile-name>","sampling":"joint","seed":42,"count":10000,"output_tokens":420}}
```

`derive_synthetic_parameters` 是统计参数反推入口，导出旧字段与联合池。报告不另写一份参数反推逻辑。
现有 `capture_frontend_prefix`、`fit_frontend_prefix`、`prefix_lineage`、`datasets/describe_traffic` 和
`workload_profile` 分别承担采集、真实 DAG 拟合、编解码、文件描述和 LRU 诊断，仍有独立用途。

## 使用

从 `rtp_llm/flexlb/tools/online_eval` 执行：

```bash
# 固定现有画像，按 SHA 定位拟合源，同时观察其他真实窗口的漂移
python3 scripts/commands/compare_traffic.py --out /tmp/fidelity --check

# 每个捕获单独重新反推参数；用于方法的样本内比较，不是 held-out 验证
python3 scripts/commands/compare_traffic.py --refit --out /tmp/fidelity-refit --check

# 快速观察或自定义对照、阈值；省略 --count 使用每个捕获的事件数
python3 scripts/commands/compare_traffic.py --profile /path/to/profile.json \
  --count 20000 --seed 42 --out /tmp/fidelity-small
```

`--captures <path>...` 选择对照文件。拟合身份按 `calibration.model_sha256` 定位，文件重命名不影响匹配；
若所选窗口不包含拟合源，工具会在随库真实数据中找回它。显式空 `--captures` 只输出画像参数校验与来源缺失说明。
缺少匹配捕获时，结论为 `UNASSESSED`；`--check` 对来源缺失或任一画像声明不一致退出 2。
报告展示实际标签的前 8 块家族聚类、首次出现冷比例、ECDF、联合密度、KS、joint TV、相关系数与完整参数。
JSON 保留精确直方计数、未舍入标量、seed、画像和捕获 SHA。相同输入路径与参数输出字节一致；HTML 无外链依赖。

画像审计复用唯一参数反推实现，检查源 SHA、provenance、所有反推参数、targets 及可用的 fit-report digest，
不会只打印失败却仍退出 0。正常单测发现与 `.github/workflows/traffic-fidelity.yml` 都运行该审计。
手写、未声称来自真实捕获的合成形状不受经验参数等值检查。

## 指标与结论的边界

共享深度从生产输出的标签与此前所有请求最长公共前缀重算；使用压缩 radix trie，避免保留展开后的每个 token。
家族统计与真实侧统一按前 8 个 block 聚类。标量分位数使用 `datasets.distribution` 的 nearest-rank；
旧画像 targets 审计继续使用原反推的 floor-ICDF 口径，以保持历史声明可重算。

joint TV 使用 log₂(token) 9–21 的 32 桶和共享深度 0–2000 blocks 的 16 桶，越界归入边界桶。
图表与度量使用相同分箱，三张密度图使用同一色标。分箱距离并不检验完整复用拓扑或时序。

默认诊断阈值为 `length_ks: [0.03, 0.10]`、`depth_ks: [0.05, 0.25]`、`joint_tv: [0.15, 0.35]`：
前值为 PASS 上限，后值为 WARN 上限，超过为 FAIL。可通过 `--thresholds` 提供 JSON，
或在 HTML 中调整、导出。它们是显式诊断策略，不是经线上验证的版本门禁，也不因新方法表现而放宽。
长度 PASS 只表示长度分布通过；cache/KV/驱逐/拼车始终至少 WARN，需 trace 对照验证时序和有限容量效应。
跨窗偏差标为预期流量漂移，不能当作画像错误；未验证输出关联、时序复用和性能结论时，`held_out_validated` 保持 false。
