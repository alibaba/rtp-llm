# 默认大盘

唯一线上参考是 [Whale-LLM-New【禁止修改】](https://grafana.alibaba-inc.com/d/iZ5Q_81nz)，本次读取版本 514，共 41 个 row、703 个 panel。Master 只参考其中的 `FlexLB · 01` 至 `07`，沿用其问题导向的分区和标题；不再参考独立 FlexLB 旧大盘，也不搬入 `FlexLB · 08` 的兼容旧指标。逐项来源与导出状态见 [coverage.md](coverage.md)。

`default.json` 是原生 Prometheus Grafana JSON，所有 row 默认折叠，子面板实际放入 row.panels，展开后才加载该组查询。没有按某一次实验挑选指标。缺失数据不补零；待埋点标题说明的是源码缺口，不是这次实验流量为零。

## P / D 与 ViT

Prefill 和 Decode 分开提供吞吐/延迟/负载、PD RPC、KV、KV 分池、Host Memory KV 和 Mock 模拟质量分区；投机采样只放在 Decode。Prefill 的缓存复用、输入 TPS 不混入 Decode；Decode 的输出 TPS、生成 Batch、输出长度不混入 Prefill。每条 Engine 查询显式使用 `role="prefill"` 或 `role="decode"`，再叠加全局 Engine 地址筛选。Mock 专有指标单列，不能当作名字相近的 real 指标。

线上 Engine 查询主要使用全局 `hippo_role`，并非每个面板都硬编码了 Prefill/Decode。源码语义、线上 PD 分离分区和局部业务过滤共同决定本地归属；Master 的 `role=PREFILL/DECODE/PDFUSION` 是被观察 Worker 的角色，不是 Master 自己的角色，保留原查询的具体限制，不能替换为 Engine 自报的小写 role。

ViT row 包括 QPS、成功/错误、下载/预处理、两级队列、Embedding 耗时、缓存和 CUDA Graph 等关键观测。当前新 mock-framework 没有 ViT 组件和导出链路，因此是待埋点面板。旧 `flexlb-mock-engine/MockVitMonitor` 确有部分 `py_rtp_vit_*` 上报，但不在新框架的启动链路里，不能标成新框架已支持。

## 已实现与缺口

Engine 的 waiting/running、Device 可用 Tokens、Device/Host 容量 Blocks、Device 不可用比例，以及 P/D TPS、终态 QPS 已接入原 RTP 指标名称及对应原面板。使用率为 0–100；used Blocks=total-free，不可用比例=(total-available)/total，不能混淆。Host 暂缺 available 与其引用分类，相关缺口保留。

真实事件的辅助 Counter/Histogram 使用 rtp_llm_*，包括接纳/终态、缓存查找/命中、Forward Tokens 和实际耗时分布；这些是本地 Prometheus 增补，不声称 real 已有完全相同的 Counter/Histogram 名称。Python 接口同步消费新名字。只有计划 Forward 保留 mock_ 前缀。实际观测归回 P/D 的 RTP row，Mock 模拟质量各保留实际/理论、偏差、模拟执行时间占比三个面板，不将最后一项当作 GPU 利用率。

TPS/QPS Gauge 使用两次采集间的真实单调时钟间隔，空闲归零；Prefill 计算 TPS 的分母只累计有计算 Token 的 Forward，含缓存 TPS 的分母只累计有输入 Token 的 Forward。窗口与 real 后台周期并非逐点一致。原始累计量不清零，门禁仍按明确时间范围查询 Counter 增量。

旧实验没有新导出，需使用其保存的原始大盘；本次不改写旧 TSDB，也不生成旧名字别名来伪造新实验结果。

仍缺逐请求等待/首 Token/总耗时、逐批 Batch 形状、PD 传输延迟/字节/失败、KV 分池引用/淘汰/Host 命中读写等细分埋点。大盘保留线上相应关键指标的入口，但不拿 running 请求数冒充 batch size，不拿 Device 命中冒充 Host 命中，不拿 Fetch 等待冒充 compute 或传输耗时。实际/理论 Forward 最近值和 Histogram 是不同口径，分开展示。

Master 的已有 Reporter 比 Engine 完整，所以可用曲线多。线上 RTP-LLM 大盘还包含本分支尚未导出的 Master 新指标、Frontend 和 Pod 监控，这些明确标为待埋点；原生 Micrometer JVM 资源另外补充，不能把 KMonitor 的不同指标名当成本地已导出。

## 共享 JVM 的资源归属

| 观测 | 当前归属 | 对齐方式 |
|---|---|---|
| 请求/队列、KV 逻辑容量、Forward | 逻辑 Engine | Engine 自有状态和独立 registry，按 Engine 地址与 role 筛选 |
| CPU、RSS、堆、GC、线程、进程 FD | Engine Host JVM | 应由宿主单独暴露一次，Prometheus instance 使用宿主管理端点；不能挂到每个 Engine 的 metrics endpoint 重复采集 |
| Pod CPU/内存限额与利用率 | 容器 | Whale 一 Pod 一 Engine 时可直接关联；本地共 JVM 时只代表整个宿主，不能均分后冒充实例消耗 |
| GPU 利用率、显存、Tensor/Core 活跃率 | 真实 GPU | CPU Mock 不产生这些硬件观测；保留待接采集入口，不合成数值 |

大盘的 Engine 宿主资源 row 暂不绑定逐 Engine 查询，避免把同一 JVM 的资源复制 N 次。后续资源接线需要 Runner 将宿主管理端点注册为独立 scrape target，一 JVM 一目标；复用 Prometheus 原生 job/instance，不增加 node/JVM 身份 tag。若必须测单个 Engine 的真实进程资源，应使用一个 JVM 一个 Engine 的运行形态。线程 CPU 只能算被跟踪线程的部分工作，堆内存分配也不是 retained RSS，不作为自动拆账方案。

## 筛选、标签与事件

Engine、Master、Client 地址筛选各自作用于对应上报者；普通数量/状态按实例平均，明确的集群速率保留总量，Histogram 分位数合并 bucket，不能平均实例 P99。Master 上报中的 engineIp 是被观察 Worker，不能与上报者 instance 混用。

本地大盘直接使用 Prometheus 的 `instance`（IP:port）作为上报者身份，不在 Java 指标中新增或改写 `host_ip/container_ip`。KMonitor 的 `[[tag_…]]` 图例转换成 Prometheus `{{label}}`，业务维度按查询实际保留。

Prefill、Decode、ViT 各有独立的地址选择器，地址来自保存指标的 role 标签，不按 IP 推测角色；没有该角色数据时隐藏对应选择器。三者分别支持多选：`All · 聚合全部` 按角色聚合；选择一个或多个具体地址则按实例分组；“逐实例全选”显式选中所有地址，仍按实例展示。全选地址与 All 是两种不同意图，不自动互相转换。未选择任何地址时回到聚合全部。Master、Client 的自建面板也支持同样的地址多选切换；平均、总量、成功率和 Histogram 的原有计算口径不变。

隐藏变量 `prefill_group`、`decode_group`、`vit_group` 分别依赖同角色的 `*_instance`：All 的值为 `.*` 时返回 `role`，具体地址返回 `instance`。Engine 查询在 `by(...)` 与图例中使用对应角色的分组变量，保留 le、state 等业务维度。跨 P/D 的 Worker 面板拆成两组独立 target，保持两侧选择互不影响。变量查询仅对 `vector(1)` 操作，不扫描业务数据，不新增上报标签；报告页和独立 Grafana 使用同一规则。Client 自建面板使用 client_group（All=role，具体地址=instance）；Master JVM 补充面板使用 master_group（All=application，具体地址=instance）。JVM 内存先按 instance/application/area 汇总池，再按当前分组平均，不能先跨 JVM 求和。以后迁入面板时必须同时适配查询分组与图例。采集完整性的 up、采集时长、样本数始终逐目标展示，以免平均值隐藏失败目标。


PromQL 中地址变量使用 `${prefill_instance}` / `${decode_instance}` / `${vit_instance}` / `${master_instance}` / `${client_instance}`，由 Prometheus 数据源完成正则与字符串转义。显式 `:regex` 格式化会绕开数据源的字符串转义，含点号的具体 IP 可能产生非法 `\.`；只测试 All 的 `.*` 无法发现这个错误。查看服务也会规范化旧报告中这些受管变量及旧版 engine_instance 的旧写法，原始报告不被修改。

标签沿用物理来源 HIPPO_APP/HIPPO_ROLE/HIPPO_SERVICE_NAME/HIPPO_SLAVE_IP/RequestedIP；本地不冒充 Pod IP。Engine 只支持 DP rank 0。业务指标与 Mock 专有指标保留原名，Prometheus 与线上 KMonitor 查询协议仍需分别适配，不能承诺直接导入同一查询 JSON。

View 的 `annotation_steps` 只配置默认选择，省略则全部显示，空数组则全部隐藏；页面可随时全选、清空或逐项选择。Grafana 的“Step / 门禁”总开关隐藏，标注保持启用，仅由 Step 选择控制是否显示。每个 Step 一个标注，完整执行记录与门禁卡片不被筛掉。门禁判定来自已保存结果，筛选、展开 row 和切换窗口都不重新发压或判定。
