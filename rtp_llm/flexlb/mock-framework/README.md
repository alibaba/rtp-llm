# Mock框架

Python负责case配置编译和执行组织，Java负责组件共享接口与实现。配置编译与本地执行器独立：compile不产生运行副作用；run通过启动配方管理进程，调用Java组件接口并执行门禁。RTP节点支持状态RPC、设备缓存、本地单token请求与Prometheus抓取；P/D链路、多JVM Client播放、历史指标存储和正式历史查看服务尚未接入，不由测试组件模拟验收。

## 目录与构建

- `python/mock_framework/config/`：YAML读取、配置检查、点分参数展开、步骤编译。
- `java/interfaces/`：组件接口的唯一权威定义、公共类型和清单生成。
- `python/mock_framework/execution/`：本地进程初始化、顺序执行、HTTP调用、门禁比较、收尾和运行记录。
- `java/runtime/`：将接口实现绑定为HTTP端点，按接口清单校验参数与回执，提供单次调用上下文。
- `java/shared/compute/`：共享Forward batch、公式绑定、确定性噪声与可取消计时执行。
- `java/shared/metrics/`：共享Engine指标定义、独立Prometheus注册表与标准文本导出。
- `java/engine/rtp/`：RTP节点入口、原生配置读取、KV块租约和生产proto状态RPC。
- `tests/`：编译与跨语言接口扩展验证；Java测试跟随模块。
- `scripts/`：构建和验证入口。

使用Java21、仓库已有的Maven Wrapper和uv：

```sh
export JAVA_HOME=<Java21目录>
./scripts/check.sh
```

`build.sh`编译Java并运行测试，然后生成`build/interfaces.json`。Python没有另一份组件接口枚举；`runtime/wait`由Python定义为执行器内置动作，不制造Java端点。生成目录不入库；Python依赖锁定在`uv.lock`。Java构建直接构建生产`flexlb-common`和`flexlb-grpc`，复用实际公式与生成协议，不继承旧Mock或online_eval构建。新组件只保留所用依赖，不引入生产模块的Spring等传递运行依赖。

## 编译case

```sh
uv run --locked mock-case compile <case目录>/case.yaml \
  --catalog build/interfaces.json \
  --shared-root <共享资源根目录> \
  --output <输出目录>/compiled.json
```

私有相对路径基于case目录解析；显式提供共享根时，`data/`、`datasets/`、`views/`路径以共享根解析。编译只解析路径，不检查数据内容或启动查询。标准输入由Java Client读取，不提供旧xz/JSONL兼容、转换器或生成器。

根字段为`id`、可选`name`、`describe`、`environment`、`phases`和可选`view`。phase顺序执行，每个step只包含`call`或`check`。动作允许`run`和`within`，读取必须提供`expect`。每个phase必须有正的`timeout`。标识符全局唯一，引用只能指向已完成定义的前置步骤及其声明返回字段，不允许向前引用。

组件使用`deployment/config`分层；Engine首版仅`rtp`。Client的`deployment.replicas`表示JVM数量，编译不把总QPS乘以副本数。资源限制实际落实由后续启动器负责，不因配置编译通过而宣称已执行。

`config`保留原生字段，点分Mock键展开为对象；只展开本层键，不改写原生JSON内容。重复键、YAML merge、循环引用、点分路径冲突均报错。`FLEXLB_CONFIG`保留原始JSON文本，编译检查JSON对象及重复键，原生业务字段是否合法仍由Master自己的加载器检查。

## 指标和断言

TTFT使用`client/requests/ttft`，参数为`stat: avg/max/p99`及`range`。时间范围保留为`all`、`latest`或负偏移区间，编译时不转换为墙上时间；接口元数据决定接受哪些形式，后端仍需明确支持具体统计组合。`all`必须位于唯一一次`client/traffic/run`之后，排空后再次run仍报错。

`prometheus/query`只接受`promql`，不接受外层`range`；编译不解析PromQL，其执行时刻、查询结果和报告点标注由后续执行器记录。普通语义接口不接受原生查询字段。

`expect`支持`eq`、`count`或`gte/lte`上下界。单位为秒的接口允许`2s`这类阈值；集合引用按Java声明类型校验。Master逐实例的Engine集合可与操作回执的Engine集合引用比较，不能通过合并各Master结果掩盖不一致。compile仅编译断言，真正的比较逻辑由执行器实现。

播放配置支持trace时间节奏或uniform总QPS，模式专属字段不混用。输出长度调整仅支持固定整数、均匀分布和正态分布，调用时实际采样、播放和优先级语义归Java Client。Client尚未接入，输出分布当前只编译与校验，不生成随机样本或启动请求。

### 输出长度调整

`input.prepare.output_tokens`可为非负整数，也可为带`distribution`的对象。类型、参数和约束在Java `OutputTokens`定义，Python读取生成清单，不维护第二份分布参数表。

```yaml
# 固定值
output_tokens: 128

# 整数均匀分布，包含min/max两个端点
output_tokens: {distribution: uniform, min: 64, max: 256, seed: 42}

# 正态分布：mean/stddev是原始分布参数，单位为token
output_tokens: {distribution: normal, mean: 128, stddev: 32, min: 1, max: 512, seed: 42}
```

随机形式必须提供int64的seed。token数与上下界是非负int32，min≤max；正态先裁剪到[min,max]，再四舍五入到整数，不进行条件重采样，因此配置的mean不是裁剪后样本均值。stddev可为0，数值必须有限。

Java Client实现时应基于seed和稳定逻辑请求身份生成每个请求的一次输出目标，不随JVM数、分片或线程调度顺序改变；同一次请求的重试保留已选目标。覆盖只影响本次输出，不改写数据集settings/observed或输入结构，采样后的实际目标随请求记录保存。此处规定采样语义，实际采样器与多JVM读取在Client批次实现。

## 共享Compute

```yaml
environment:
  compute:
    prefill:
      expression: "param(baseMs, 1) + param(tokenMs, 0.01)*totalComputeTokens + 1e-8*sum(computeTokens^2)"
      noise:
        scale: 1
        stddev_ratio: 0.05
        max_deviation_ms: 20
        seed: 42
    decode:
      expression: "param(baseMs, 1) + param(perRequestMs, 0.1)*batchSize"
```

Compute按prefill/decode/vit分别配置，不复制引擎调度、缓存或KV容量参数。字段类型在Java`ComputeSpec`定义，随接口清单生成配置schema；Python不维护第二份字段表。Python检查结构与数值类型，Java组件初始化时解析公式、检查噪声约束并补全默认值；运行时仍检查每个真实batch的公式结果。没有旧标定文件或内置模型作为静默回退，未配置的角色不能计算。

`ForwardBatch`保留各DP分组及完整请求长度、已匹配cache、目标输出长度、已生成长度、本轮stepTokens、优先级和图片原始shape。Engine自己构造batch、计算cache、处理图片和推进请求状态。`ForwardModel`只接收batch并返回基础Duration；默认`FormulaForward`当前只支持一个非空DP，多DP明确报错，不擅自求和或取最大值。其他模型实现可以直接读取完整输入，不必修改公共输入类型。

公式复用生产`ArithmeticFormula`的运算、函数、`param(name, value)`和`sum(...)`。常量参数直接写在表达式里，不另设parameters映射；param初始值遵循生产工具的解析时求值语义，应使用常量，不引用运行时batch变量。式子结果单位为毫秒，保留小数至纳秒精度，不隐式加1ms下限。未知变量、非法作用域、负值、非有限值和耗时溢出均报错。

逐请求变量包括inputTokens、hitCacheTokens、computeTokens、hasHitCache、outputTokens、generatedTokens、contextTokens、stepTokens、priority、imageCount、imagePixels。它们只在`sum(...)`中绑定；batch层使用batchSize、dpRank及自动派生的total/max变量，例如totalComputeTokens、maxInputTokens。`totalComputeTokens^2`与`sum(computeTokens^2)`含义不同，保留实际请求分布。`computeTokens=inputTokens-hitCacheTokens`、`contextTokens=inputTokens+generatedTokens`；stepTokens直接取Engine提供值，不由共享层推断batch如何形成。图片仅保留原始shape及按公式需要使用的像素数，不计算预处理或特征token数。

噪声先按scale缩放基础耗时，再加高斯扰动；scale默认1，stddev_ms/stddev_ratio默认0，两者按平方和开根组合标准差。启用高斯时必须显式提供seed和正的max_deviation_ms；扰动裁剪到该上下界，最终耗时下限为0。一次Forward只取样一次。随机数由seed、Engine身份、角色与batch身份决定，不依赖调用顺序或共享随机游标。不同batch必须提供不同的实际调用身份；同一身份重算得到同一计划耗时。

组件通过`ComputeModels.fromComponentFile`读取Runner生成的配置文件，通过`effectiveConfiguration()`取得含默认值的配置用于保存。`plan`只计算Sample，`SimulatedCompute.submit`才启动等待，返回Execution；Engine通过`Execution.cancel()`取消，completion返回唯一的COMPLETED或CANCELLED结果及实际elapsedNanos。计时器关闭会取消所有未完成计算并拒绝新任务，业务回调在独立虚拟线程执行，不阻塞其他计时或取消。共享层不提交cache、不释放Engine资源，也不把计算完成当作请求成功；这些动作由Engine依据真实结果执行。

Compute已接入RTP本地单token链路；完成Forward时记录实际计时和token数。组件测试不作为完整Engine行为、TPS或压测验收。

## 扩展接口

在`MasterApi`、`EngineApi`或`ClientApi`添加方法，使用`@Endpoint`声明接口名、动作/读取类别和可选单位；使用Java record声明参数，`@Argument`说明可选性和范围。引用类型直接从record类型生成，不再维护另一份类型名称。实现放所属组件，不放Python。重新构建即生成包含新方法的清单。

编译器根据清单校验参数和返回引用，不为新接口修改parser/compiler。Java方法必须有一个record参数，即使无参数也使用`Types.None`；同名接口和不支持的类型在生成清单时失败。完全新增一组Java API时在`InterfaceRegistry.standard()`登记其接口类。

跨语言扩展测试覆盖Java新增方法→生成参数和返回类型→Python直接编译调用及错误类型拒绝；它只验证接口扩展和配置编译，不替代网络调用、Engine行为或性能验收。

## 本地执行底座

```sh
uv run --locked mock-case run <case目录>/case.yaml \
  --catalog build/interfaces.json \
  --launch <启动配方.yaml> \
  --output <新的运行目录>
```

启动配方是开发接线配置，不放入case业务区域，也不新增一套业务接口定义。RTP节点使用下述独立JAR入口；Master/Client完整业务启动配方尚未提供。

```yaml
components:
  master:
    command: [<Java21可执行文件>, -cp, <组件classpath>, <组件入口类>]
    cwd: <组件工作目录>
    startup_timeout: 30s
    env: {}
bindings:
  master/: master-0
```

配方为environment中每类组件启动replicas个独立进程，编号为`<组件>-<从0开始的序号>`；compute作为配置传入，不独立启动。command是argv数组，不经过shell。cwd相对启动配方解析。组件从`MOCK_CONFIG_FILE`读取本实例配置和完整environment；`MOCK_NODE_ID`、`MOCK_COMPONENT`、`MOCK_OUTPUT_DIR`提供实例身份、角色与输出目录。`MOCK_READY_FILE`由组件初始化并绑定API后，通过`ApiServer.publishReady`原子发布，端口由系统分配。启动超时或提前退出都失败并回收已启动进程。

bindings按最长接口前缀指定处理该接口的进程，运行前检查每个步骤都有绑定。该映射指定一个接口实现的入口，不执行副本指标求和或流量分片；Client组的协调和聚合由后续Client实现承担。当前本地启动器不落实CPU、内存或heap限制，配置这些限制时直接拒绝执行，避免把未落实的资源配置当作已生效。

Java组件初始化后调用`new ApiServer(address).bind(MasterApi.class, implementation).start()`，Engine和Client同理。服务器依据同一Java接口生成的schema检查参数与返回值；Optional参数缺省传null，非法值不会通过Jackson自动转换成另一种类型。组件可在方法内读取`InvocationContext.current()`，取得run模式、固定queryTime、解析后的观测范围和超时预算。后台模式仍须由组件启动实际活动并返回启动回执，HTTP服务器不替组件虚构后台执行。接口实现处理业务，Python处理expect。

所有步骤按顺序执行。phase使用单调时钟计算总预算，call的within不能超过剩余阶段时间。进程退出、调用异常、空查询或门禁失败都会中止后续步骤并收尾。动作不自动重试；超时记录结果可能未知。收尾对当前运行拥有的进程组先TERM、再KILL，主动取消同样执行收尾。这是进程回收，业务Client的stop/drain、请求记录刷盘和采集器收尾需在对应组件接入时实现。

每次读取固定query_time；`latest`记录查询点，`[-60s, 0s]`解析为明确起止时间，`all`取唯一发流回执中的started_at到当前查询时刻。原始PromQL不解析内部窗口，记录单点查询和全部返回样本，所有样本必须满足expect，空结果为ERROR。集合eq忽略顺序，拒绝重复身份；Master的Engine集合逐个比较，空Master结果不能通过门禁。

运行目录不覆盖已有目录，保存原始case、编译配置、启动配方、接口清单、代码版本与工作区差异、组件配置和日志、phase/step时间线、门禁实际值及范围、最终结果。PASS表示所执行步骤通过，FAIL表示门禁不满足，ERROR表示执行或数据异常，CANCELLED表示取消；清理异常不能得到PASS。RTP节点已可导出本地链路指标，但运行器尚未管理Prometheus历史存储，也未生成逐请求记录；正式报告的数据保留由后续采集接入完成。测试中的TransportProbe只用于验证真实JVM进程及网络链路，不实现Engine或压测行为。

## RTP节点与缓存底座

构建生成`java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar`，包含生产生成的gRPC/protobuf类型及新组件所需运行依赖。启动配方中使用`command: [<Java21>, -jar, <该JAR绝对路径>]`，组件可为prefill或decode；每个副本独立JVM，端口由系统分配，当前仅监听本机回环地址。

```yaml
environment:
  prefill:
    engine: rtp
    config:
      SEQ_SIZE_PER_BLOCK: "64"
      MAX_SEQ_LEN: "32768"
      MAX_BATCH_TOKENS_SIZE: "8192"
      REUSE_CACHE: "true"
      capacity.device_blocks: 1024
      capacity.reserve_blocks: 52
  compute:
    prefill:
      expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生字段支持server_args的小写名称或Whale环境变量的大写名称；同一字段两种写法同时存在时报错，不做静默优先级覆盖。整数/布尔字段接受对应原生env文本，保留原始业务配置，不展开其中的JSON文本。当前读取seq_size_per_block、max_seq_len、max_batch_tokens_size、reuse_cache、enable_device_cache、reserve_block_ratio、tp_size、dp_size、dp_rank；其他原生内容保留但不宣称已实现其业务效果。block size和max seq length必须显式配置，不从虚拟GPU型号或真实模型结构猜测；当前仅支持dp_size=1、dp_rank=0。

Mock容量通过点分键capacity.device_blocks配置，单位为本实例的物理块；capacity.reserve_blocks可提供观测到的绝对预留块数。未提供绝对预留值时，按原生reserve_block_ratio计算，默认5%，向上取整。总token容量=物理块数×原生block size，不把部署内存作为模拟KV容量。默认参数和实际解析值写入组件目录的effective-config.json，原始业务配置仍保留；复制配置不表示忽略的原生功能已完成对齐。

节点在原生gRPC提供CheckHealth、GetWorkerStatus和GetCacheStatus，P节点额外支持下述本地GenerateStreamCall子集。健康返回OK；WorkerStatus使用RoleType.PREFILL/DECODE和对应类型枚举，available_concurrency保持生产默认0，KV容量使用token单位；CacheStatus另外提供block size。缓存键版本只随已发布键集合变化，同版本省略键列表，但每次仍返回当前可用容量。precision暂为FP16的Mock状态标记，不模拟量化或真实模型布局。

设备块池按引用管理共享前缀：可用容量包含空闲块和未被引用的LRU块，物理free不包含LRU数据。分配先引用命中、再检查净新增需求与reserve，失败回滚引用；需要空闲物理块时才淘汰LRU。增长失败保留已增长的部分，最终由原owner释放。DeviceCache封装原始池租约，完成、取消和关闭只有一次所有权转换；publish可保留已计算P数据供连接器使用，但不会自动推进P/D协议。取消后不能再次发布，关闭清除缓存并拒绝新分配。

组件目录同时保存endpoints.json，记录本实例原生RPC地址、控制地址、支持的RPC和本地生成范围。HTTP /health仅报告已绑定的公共业务接口清单；当前RTP节点该清单为空，不把原生状态RPC或健康探测扩充为case业务接口。Enqueue、Fetch、原生Cancel、RemoteGenerate和P/D接续尚未绑定，调用返回gRPC UNIMPLEMENTED；EngineApi动作/指标查询适配尚未提供，Prometheus原始抓取见下文。节点启动通过不表示压测或完整RTP行为通过。

### 本地单token请求

P节点接受文本原生token_ids、max_new_tokens=1的GenerateStreamCall。生产PrefillRpcServer对这种请求同样走LocalRpcServer，不启用P/D；即使can_use_pd_separation为true也不会走D。该路径使用FIFO、每轮一个请求、max_context_batch_size=1；显式配置更大context batch时启动报错，不默认为已支持batch。多token、prefill-only零输出、图片、adapter、自定义unique_key、beam/多返回和额外logits/hidden-state等输出模式尚未实现，明确拒绝，不套固定成功帧。

LocalPrefill管理请求与状态，PrefixKeys计算原生文本前缀，LocalResponse只负责响应编码。请求实际经过排队→KV分配→共享Compute估时及等待→原生输出→释放；排队阶段不占KV，选中时才匹配和分配。命中遵守全局与单请求reuse开关，不能复用最后一个prompt token；只索引完整块，不发布partial block。只有成功结束且全局/请求reuse与device cache均启用时才写回缓存。

结果是一帧finished输出，Tensor为little-endian INT32、shape=[1,1,1]。token值重复输入末token，是模拟载荷，不做真实模型推理或采样。aux_info包含实际输入长度、命中长度和输出长度；等待由共享公式和噪声决定，不用固定响应绕过计算。

请求timeout_ms从接收开始独立计时，覆盖排队和执行，不随阶段重置；RPC deadline和流取消也会终止对应Session。回调持有Session身份，延迟取消不能作用于后来复用相同request_id的请求。完成、取消、超时和关闭在同一所有权边界处理一次，释放后不能再次提交cache。602/603/605通过原生ErrorDetailsPB尾部元数据返回；不自动重试资源分配，也不因取消而发送成功帧。

WorkerStatus返回实际waiting/running任务、队列数、finished任务与latest_finished_version。当前Mock入口固定最多1024个活动请求；完成历史也保留1024条，调用方请求的旧版本已丢失时返回OUT_OF_RANGE，避免把缺失历史当作完整结果。该上限是本地Mock保护，不宣称是real Engine的调度规则。此实现不含动态batch、P/D资源准备/接续、GPU、完整重试策略或压力能力验收。

### 共享Engine指标

`endpoints.json`的`metrics`给出Prometheus抓取URL，与控制接口共用HTTP端口，仅接受`GET /metrics`。使用Prometheus官方Java simpleclient 0.16.0及TextFormat导出标准0.0.4文本。每个节点有独立注册表，不使用默认全局注册表；没有常驻采样线程，累计量在事件发生时更新，瞬时量在抓取时读取。

EngineMetrics放在共享目录，不依赖RTP协议、缓存实现或调度器。其他Engine使用同一组件，传入事件和当前状态即可。固定标签为`engine`、`role`、`node`，额外维度仅为下表枚举，不把请求ID、cache key或异常文本放进标签。

| 指标 | 类型与额外标签 | 上报时机与含义 |
| --- | --- | --- |
| `mock_engine_requests_started_total` | Counter | 校验并接纳请求后加一；不支持的协议、重复ID和入口超限不计入。 |
| `mock_engine_requests_finished_total` | Counter，`outcome=success/cancelled/timeout/error` | 已接纳请求只记录一次终态；节点关闭计error。这是Engine结果，不能替代Client最终成功率。 |
| `mock_engine_cache_lookup_tokens_total` | Counter | 成功获得KV租约时累计输入token数，包括已命中的部分。 |
| `mock_engine_cache_hit_tokens_total` | Counter | 同一次分配中实际复用的历史前缀token数；后续请求取消不撤销已发生的查找。 |
| `mock_engine_forward_tokens_total` | Counter，`kind=input/compute/output` | 活跃请求的Forward完成时记录输入、实际计算和输出token数。当前P路径分别为输入长度、输入减命中、1；被取消或超时的Forward不补记成功计算。 |
| `mock_engine_forward_seconds` | Histogram | 完成Forward的实际计时，单位秒，排除请求排队时间；提供bucket、sum、count。它不是Client TTFT。 |
| `mock_engine_requests` | Gauge，`state=waiting/running` | 当前排队、执行数量。 |
| `mock_engine_kv_tokens` | Gauge，`state=total/available/free/held` | 当前KV容量，单位token。available含可淘汰的未引用缓存，不能当作物理free。 |

累计量预建为0，抓取不会清零。缓存命中率按hit tokens增量除以lookup tokens增量做token加权，不能平均每请求命中率。Prefill计算TPS可由完成的compute tokens增量除以Forward seconds sum增量得到，带cache TPS使用input tokens；若要墙钟吞吐则取token counter的rate，两者分母不同。共享层只保存观测值，由查询接口选择对应口径。分母为0时没有可判断的比率，不补成100%或0来通过门禁。

这些指标是新框架的公共口径，不冒充KMonitor原指标。后续语义查询接口负责PromQL，case仍优先调用接口；Grafana旧大盘的指标映射、Runner管理Prometheus与历史保留、Client请求指标尚未接入。打包测试保存一次metrics.prom供核对，不代表历史采集完成。
