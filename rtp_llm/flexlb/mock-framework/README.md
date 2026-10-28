# Mock框架

Python负责case配置编译和执行组织，Java负责组件共享接口与实现。配置编译与本地执行器独立：compile不产生运行副作用；run通过启动配方管理进程，调用Java组件接口并执行门禁。RTP节点支持状态RPC、设备缓存、文本P/D生成、共享Decode推进及停止模型、Prometheus抓取与指标查询；Runner可管理本次运行的Prometheus及历史存储。真实Master、多JVM Client播放和正式历史查看服务尚未接入，不由测试组件模拟验收。

## 目录与构建

- `python/mock_framework/config/`：YAML读取、配置检查、点分参数展开、步骤编译。
- `java/interfaces/`：组件接口的唯一权威定义、公共类型和清单生成。
- `python/mock_framework/execution/`：本地进程初始化、顺序执行、HTTP调用、门禁比较、收尾和运行记录。
- `java/runtime/`：将接口实现绑定为HTTP端点，按接口清单校验参数与回执，提供单次调用上下文。
- `java/shared/compute/`：共享Forward batch、公式绑定、确定性噪声、Decode推进、停止模型与可取消计时执行。
- `java/shared/metrics/`：共享Engine指标定义、独立注册表、Prometheus导出与查询实现。
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
  prefill:
    engine: rtp
    config:
      compute.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
      compute.noise.scale: 1
      compute.noise.stddev_ratio: 0.05
      compute.noise.max_deviation_ms: 20
      compute.noise.seed: 42
      generation.eos_probability: 0.01
      generation.seed: 42
  decode:
    engine: rtp
    config:
      compute.expression: "1 + 0.1*totalStepTokens + 0.001*totalContextTokens"
      decode.draft_tokens: 3
      decode.acceptance_rate: 0.8
      decode.seed: 42
```

各组件组在自己的config中配置一份Compute公式和噪声，不再提供顶层environment.compute，也没有组内/顶层覆盖优先级；旧位置直接报错。Java仍共用同一个ComputeModel实现，配置位置不影响组件复用。配置schema由Java的ComputeSpec、DecodeSpec、StoppingSpec生成，Python不另写字段表。Python校验结构、类型和概率边界，Java初始化时解析公式、补默认值并检查随机模型必需的seed；运行时检查真实batch的计算结果。RTP启动必须提供本组公式，不从其他组或旧标定文件静默回退。

`ForwardBatch`保留各DP分组及完整请求长度、已匹配cache、目标输出长度、已生成长度、本轮stepTokens、优先级和图片原始shape。Engine自己构造batch、计算cache、处理图片和推进请求状态。`ForwardModel`只接收batch并返回基础Duration；默认`FormulaForward`当前只支持一个非空DP，多DP明确报错，不擅自求和或取最大值。其他模型实现可以直接读取完整输入，不必修改公共输入类型。

公式复用生产`ArithmeticFormula`的运算、函数、`param(name, value)`和`sum(...)`。常量参数直接写在表达式里，不另设parameters映射；param初始值遵循生产工具的解析时求值语义，应使用常量，不引用运行时batch变量。式子结果单位为毫秒，保留小数至纳秒精度，不隐式加1ms下限。未知变量、非法作用域、负值、非有限值和耗时溢出均报错。

逐请求变量包括inputTokens、hitCacheTokens、computeTokens、hasHitCache、outputTokens、generatedTokens、contextTokens、stepTokens、priority、imageCount、imagePixels。它们只在`sum(...)`中绑定；batch层使用batchSize、dpRank及自动派生的total/max变量，例如totalComputeTokens、maxInputTokens。`totalComputeTokens^2`与`sum(computeTokens^2)`含义不同，保留实际请求分布。`computeTokens=inputTokens-hitCacheTokens`、`contextTokens=inputTokens+generatedTokens`；stepTokens直接取Engine提供值，不由共享层推断batch如何形成。图片仅保留原始shape及按公式需要使用的像素数，不计算预处理或特征token数。

噪声先按scale缩放基础耗时，再加高斯扰动；scale默认1，stddev_ms/stddev_ratio默认0，两者按平方和开根组合标准差。启用高斯时必须显式提供seed和正的max_deviation_ms；扰动裁剪到该上下界，最终耗时下限为0。一次Forward只取样一次。随机数由seed、Engine身份、角色与batch身份决定，不依赖调用顺序或共享随机游标。不同batch必须提供不同的实际调用身份；同一身份重算得到同一计划耗时。

组件通过`ComputeModel.fromComponentFile`读取Runner生成的本组配置，通过`effectiveConfiguration()`取得含默认值的配置用于保存。`plan`只计算Sample，`SimulatedCompute.submit`才启动等待，返回Execution；Engine通过`Execution.cancel()`取消，completion返回唯一的COMPLETED或CANCELLED结果及实际elapsedNanos。计时器关闭会取消所有未完成计算并拒绝新任务，业务回调在独立虚拟线程执行，不阻塞其他计时或取消。共享层不提交cache、不释放Engine资源，也不把计算完成当作请求成功；这些动作由Engine依据真实结果执行。

Compute已接入RTP Prefill和Decode链路；完成Forward时记录实际计时和token数。组件测试不作为完整Engine行为、TPS或压测验收。

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

所有步骤按顺序执行。phase使用单调时钟计算总预算，call的within不能超过剩余阶段时间。进程退出、调用异常、空查询或门禁失败都会中止后续步骤并收尾。动作不自动重试；超时记录结果可能未知。收尾对当前运行拥有的进程组先TERM、再KILL，主动取消同样执行收尾。托管Prometheus按下述顺序采集并退出；业务Client的stop/drain与请求记录刷盘仍需在Client接入时实现。

每次读取固定query_time；`latest`记录查询点，`[-60s, 0s]`解析为明确起止时间，`all`取唯一发流回执中的started_at到当前查询时刻。原始PromQL不解析内部窗口，记录单点查询和全部返回样本，所有样本必须满足expect，空结果为ERROR。集合eq忽略顺序，拒绝重复身份；Master的Engine集合逐个比较，空Master结果不能通过门禁。

运行目录不覆盖已有目录，保存原始case、编译配置、启动配方、接口清单、代码版本与工作区差异、组件配置和日志、phase/step时间线、门禁实际值及范围、最终结果。PASS表示所执行步骤通过，FAIL表示门禁不满足，ERROR表示执行或数据异常，CANCELLED表示取消；清理异常不能得到PASS。配置托管采集时同时保留Prometheus配置、实际版本、目标、日志和TSDB/WAL。逐请求记录与正式历史查看服务尚未接入。测试中的TransportProbe只用于验证真实JVM进程及网络链路，不实现Engine或压测行为。

### Runner托管Prometheus

采集器属于执行设置，放在启动配方中，case顶层不新增字段：

```yaml
prometheus:
  binary: /path/to/prometheus
  scrape_interval: 1s
  startup_timeout: 30s
```

binary必须是已提供的可执行文件，相对路径按启动配方目录解析；Runner不下载或安装。scrape_interval默认1s，支持整数毫秒；startup_timeout默认30s，分别约束采集器启动和基线等待。当前验证版本为Prometheus3.5.0，需支持JSON日志和系统分配监听端口。

Runner先在新运行目录的prometheus/下生成配置与空file discovery文件，启动独立进程。Prometheus绑定127.0.0.1:0，Runner从该进程的结构化日志读取实际端口，不预占或猜测端口。组件实际config.json自动注入prometheus.url和job；原始case及编译配置不改写。托管模式下不能再显式配置组件prometheus参数，冲突在启动前报错。

各组件ready后必须在endpoints.json提供匹配的id和metrics URL；缺失或不合法直接报错，不能静默漏采。Runner原子更新targets.json，全部节点成功采样至少两次后才开始phase/step。当前发现的是启动配方创建的固定节点集合，运行中动态增删目标尚未接入。每次运行有独立Prometheus和TSDB，job固定为mock，不要求用户手工创建实验ID。

收尾先等待每个目标在收尾开始后完成一次成功采样，再停止组件，最后向Prometheus发TERM并等待正常退出。超时才KILL；采集器被杀、退出异常或最后采样失败会记录数据不完整并令结果ERROR，不能得到PASS。启动未完成时取消或失败，仍回收所有已拥有进程并保留现场，不宣称已采完实验。

prometheus/保留prometheus.yaml、targets.json、data/、stdout.log、stderr.log及collection.json。collection.json记录版本、实际地址、目标、基线时刻、最后采样时刻和完整性结果；complete表示本次托管边界采集与关闭成功，不替代对中间up样本或业务指标的判断。门禁FAIL与用户取消可以拥有complete=true的采集记录。TSDB不转换为JSON曲线，也不会在收尾删除；服务停止后由后续回放服务读取。数据保存采用Prometheus默认保留策略，本批不增加归档、长期保留或报告服务。

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
      TEST_BLOCK_NUM: "1025"
      RESERVE_BLOCK_RATIO: "5"
      compute.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生字段支持server_args的小写名称或Whale环境变量的大写名称；同一字段两种写法同时存在时报错，不做静默优先级覆盖。整数/布尔字段接受对应原生env文本，保留原始业务配置，不展开其中的JSON文本。当前读取seq_size_per_block、max_seq_len、max_batch_tokens_size、reuse_cache、enable_device_cache、disable_decode_first_malloc_device_reuse、load_cache_timeout_ms、reserve_block_ratio、tp_size、dp_size、dp_rank；其他原生内容保留但不宣称已实现其业务效果。block size和max seq length必须显式配置，不从虚拟GPU型号或真实模型结构猜测；当前仅支持dp_size=1、dp_rank=0。

容量沿用RTP原生test_block_num/TEST_BLOCK_NUM和reserve_block_ratio/RESERVE_BLOCK_RATIO，不再支持capacity.device_blocks或capacity.reserve_blocks。真实设备池的block 0不可分配，因此test_block_num必须至少为2，实际可用块数=test_block_num-1，总token容量=可用块数×seq_size_per_block。预留水位按可用块数×原生百分比做整数除法，向下取整，默认5%；它是首次分配的容量门禁，不阻止运行中请求使用预留进行增长。effective-config.json保存物理块数、可用块数、预留水位与模型默认值。

RTP真实实现也支持kv_cache_mem_mb，但从字节换算块数需要真实KV布局；当前Mock不加载模型层/head或GPU内存，要求显式test_block_num，不根据部署内存或虚拟GPU型号猜测。复制的其他原生配置保留，但不能据此宣称未实现的功能生效。这里只对齐单池容量语义，未实现混合KV分组的物理布局。

节点在原生gRPC提供CheckHealth、GetWorkerStatus和GetCacheStatus，P支持下述本地GenerateStreamCall子集，D支持RemoteGenerate的ALLOCATE、LOAD与GENERATE子集。健康返回OK；WorkerStatus使用RoleType.PREFILL/DECODE和对应类型枚举，available_concurrency保持生产默认0，KV容量使用token单位；CacheStatus另外提供block size。缓存键版本只随已发布键集合变化，同版本省略键列表，但每次仍返回当前可用容量。precision暂为FP16的Mock状态标记，不模拟量化或真实模型布局。

设备块池按引用管理共享前缀：可用容量包含空闲块和未被引用的LRU块，物理free不包含LRU数据。分配先引用命中、再检查净新增需求与reserve，失败回滚引用；需要空闲物理块时才淘汰LRU。增长失败保留已增长的部分，最终由原owner释放。DeviceCache封装原始池租约，完成、取消和关闭只有一次所有权转换；publish可保留已计算P数据供连接器使用，但不会自动推进P/D协议。取消后不能再次发布，关闭清除缓存并拒绝新分配。

组件目录同时保存endpoints.json，记录本实例原生RPC地址、控制地址、支持的RPC和具体阶段范围。HTTP /health仅报告已绑定的公共业务接口清单；未配置Prometheus查询时清单为空，配置后绑定下述5个只读接口，不把健康探测扩充为case业务接口。P支持GenerateStreamCall和下述Enqueue/Fetch/Cancel文本P/D链路。节点启动或ALLOCATE通过不表示完整P/D或压力验收。

### 本地单token请求

P节点接受文本原生token_ids、max_new_tokens=1的GenerateStreamCall。生产PrefillRpcServer对这种请求同样走LocalRpcServer，不启用P/D；即使can_use_pd_separation为true也不会走D。该路径使用FIFO、每轮一个请求、max_context_batch_size=1；显式配置更大context batch时启动报错，不默认为已支持batch。本地多token、prefill-only零输出、图片、adapter、自定义unique_key、beam/多返回和额外logits/hidden-state等输出模式尚未实现，明确拒绝，不套固定成功帧。

LocalPrefill管理请求与状态，PrefixKeys计算原生文本前缀，LocalResponse只负责响应编码。请求实际经过排队→KV分配→共享Compute估时及等待→原生输出→释放；排队阶段不占KV，选中时才匹配和分配。命中遵守全局与单请求reuse开关，不能复用最后一个prompt token；只索引完整块，不发布partial block。只有成功结束且全局/请求reuse与device cache均启用时才写回缓存。

结果是一帧finished输出，Tensor为little-endian INT32、shape=[1,1,1]。token值重复输入末token，是模拟载荷，不做真实模型推理或采样。aux_info包含实际输入长度、命中长度和输出长度；等待由共享公式和噪声决定，不用固定响应绕过计算。

请求timeout_ms从接收开始独立计时，覆盖排队和执行，不随阶段重置；RPC deadline和流取消也会终止对应Session。回调持有Session身份，延迟取消不能作用于后来复用相同request_id的请求。完成、取消、超时和关闭在同一所有权边界处理一次，释放后不能再次提交cache。602/603/605通过原生ErrorDetailsPB尾部元数据返回；不自动重试资源分配，也不因取消而发送成功帧。

WorkerStatus返回实际waiting/running任务、队列数、finished任务与latest_finished_version。当前Mock入口固定最多1024个活动请求；完成历史也保留1024条，调用方请求的旧版本已丢失时返回OUT_OF_RANGE，避免把缺失历史当作完整结果。该上限是本地Mock保护，不宣称是real Engine的调度规则。此本地P路径不含动态batch、GPU、完整重试策略或压力能力验收；多token文本使用下述P/D路径。

### D资源准备

D通过原生双向RemoteGenerate接受文本单序列ALLOCATE。DecodeAdmission持有请求及租约，RtpRpcService只适配原生协议；TaskHistory与RtpErrors分别统一P/D的完成版本和原生错误编码。资源身份采用client_id和request_id组合，同一身份不能覆盖已有owner，不同client可以使用相同request_id。活动数量仍受1024的Mock保护上限约束。

ACK只在实际分配成功且截止时间未到时发出。临时容量不足时等待资源变化，唤醒后只有allocation generation改变才再次申请；不使用旧的短重试次数/预算将健康等待提前判失败。永久需求超出可用物理容量时返回602。请求timeout_ms与RPC deadline取更早者，覆盖排队、等待确认及持有阶段；取消、半关闭、超时和节点关闭均释放一次，不发布尚未计算的数据。timeout_ms=0表示没有该项预算，仍受RPC取消/deadline及节点生命周期约束。

首次分配的历史前缀复用遵守全局/单请求reuse、device cache及原生disable_decode_first_malloc_device_reuse；该开关默认false，与CacheConfig.h默认一致。命中只取完整块，排除最后prompt token；已由本请求传输的KV不能冒充历史命中。LOAD得到的KV只归当前请求持有，取消不会生成新cache key。

WorkerStatus区分TASK_PHASE_RECEIVED和TASK_PHASE_KV_ALLOCATED。分配、LOAD等待及已LOAD等待生成均不报告为RUNNING；收到GENERATE后才进入Decode队列并记录实际Forward、输出token和终态。endpoints.json明确remote_generation支持ALLOCATE、LOAD、GENERATE、PREFILL_COMPLETE，load=true；ACK声明supports_prefill_completion。越序、重复LOAD或重复GENERATE失败并释放租约；GENERATE后的请求流半关闭允许继续接收输出，前置阶段半关闭仍取消请求。

### P→D元数据交接

P通过独立的Mock内部gRPC服务`mock.rtp.PrefillTransfer`提供Prepare和Read，与原生RPC共用端口，地址及metadata_only标记写入endpoints.json的mock_transfer。协议定义在RTP目录的prefill_transfer.proto，不进入case公共接口清单，也不宣称兼容真实cache-store的TCP/RDMA协议。这里只模拟请求数据就绪、块布局和资源交接，不生成或复制KV张量，不模拟传输带宽。

Prepare输入client_id、原生GenerateInputPB和block_size；它复用LocalPrefill的FIFO、历史命中、共享Compute及KV租约。max_new_tokens可以大于1，但此阶段只计算Prefill及第一个模拟token，不执行D生成。Forward完成后发出一条TransferMetadata，包含block_size、input_tokens和完整块的cache_keys，流保持打开，含partial尾块的实际KV租约继续占用。数据没有读取前，既不归还容量，也不以仅存在于LRU的key冒充活跃来源。P请求ID在本节点活动期间唯一，传输身份为client_id与request_id；调用方必须先建立Prepare，再发起D LOAD。

D的ALLOCATE.peer_addrs填写一个P的mock_transfer地址，LOAD帧沿用原生语义，不重新提交来源。D先持有自己的实际租约，再调用Read；Read核对身份、完整prompt和block size，若Forward尚未完成则等待同一个就绪结果。每个来源只接受一个Reader；读取成功交付元数据后，P完成并释放租约，Prepare流结束。D再校验元数据与本地块布局，收到完整成功响应且预算未过期后才确认LOAD；P传来的前缀不计为D历史命中，也不提前发布为可复用D缓存。

找不到活跃来源、prompt或block size不同、重复Reader、连接失败、请求超时都不会产生LOAD成功确认。load_cache_timeout_ms遵循原生名称与大写env别名，默认5秒，非正值按原生回退为5秒；同时受D请求总预算及RPC deadline限制。D在Read期间取消/超时会关闭来源连接，中断P的计算并回收两侧租约；P取消、过期或关闭也会令等待中的Read及LOAD失败。尚未建立Reader或读取被入口校验拒绝时，P仍由Prepare拥有；Prepare必须有请求timeout或RPC deadline，避免无人读取时永久占用，生产者也应在失败时取消自己的Prepare流。

该内部服务仍支持测试客户端显式Prepare/Read；正式P/D入口使用同一个PrefillTransfer注册来源，不绕回本机发起另一条Prepare RPC。内部来源由P请求所有者管理，允许无deadline请求；外部Prepare仍要求有限deadline。真实Master及正式Client尚未接入，节点间验证不替代正式case验收。

### 文本P/D自动生成

P的原生GenerateStreamCall对max_new_tokens>1且can_use_pd_separation=true的请求启用P/D，要求role_addrs中恰好一个有效DECODE地址。max_new_tokens=1仍走原有本地路径。PrefillGeneration负责整条P/D调用，LocalPrefill只负责本地Forward及交接前的KV；D在LOAD成功后由DecodeAdmission继续推进生成。节点端点的pd_generation描述这一能力，公共case接口不新增P/D内部动作。

阶段顺序依据真实PrefillRpcServer：先向D发ALLOCATE并等待ACK，再启动P计算，随后发LOAD、输出P首token、等待LOAD确认、发送GENERATE并转发D输出。首token已计入总输出长度；未启用推测时，长度N需要D执行N-1次Forward，启用后按实际接受长度推进，不重复输出首token。NativeProbe通过两个独立打包JVM验证此链路，测试客户端只向P提交一次请求。

D当前使用一个模拟执行槽，每轮轮转活动请求，尚不合并动态batch。每轮向共享Compute提供原始输入长度、历史命中、目标输出长度、已生成长度、本轮verifyTokens作为stepTokens及优先级；公式和噪声实际计时完成后才提交本轮输出。每轮前按验证上下文长度增长KV；临时不足等待allocation generation变化，不阻塞其他可执行请求，永久不足返回602，超时返回603。WorkerStatus区分等待和执行，记录执行累计时间与实际D迭代数，Prometheus每轮记录一次完成事件。

模拟token重复首token值，不执行真实词表采样或语言推理；EOS使用下述停止模型。return_incremental=true时每帧仅含本轮提交的新token，否则返回当前累计输出，aux.output_len始终为累计长度，最后一帧finished=true。P保留自己的TTFT，并分别记录P与D的历史命中；总命中沿用P口径，不把两端命中相加。暂不支持stop words、真实草稿张量/原生推测提案交接、额外位置交接、beam、多返回、多模态及额外logits等输出，均显式拒绝。当前要求input+max_new_tokens小于D的max_seq_len，超过时返回605；不宣称已实现真实引擎的长度截断与reserveStep策略。

D成功时按reuse/device开关写回完整prompt块的key并释放租约，新增生成内容暂不建立额外缓存索引。取消或失败不发布新D缓存。D成功/失败只是Engine口径，仍不替代正式Client成功率。

P请求timeout从入口开始，覆盖D预留、P计算、LOAD、D生成及慢消费者等待，与RPC deadline取较早者，不逐阶段重置。取消会关闭当前请求Context、取消P来源和D活动，节点关闭同样回收请求。节点连接池复用到各目标的通道；取消一个请求不会关闭其他请求的连接。P转发使用受控拉取和有限Inbox，P/D均遵守RPC发送就绪状态，避免为慢消费者持续积压整条输出。

原生批量入口支持EnqueueBatch和EnqueueGroup（仅DP rank 0）。每个请求先得到D的真实ALLOCATE确认，再提交P的Forward队列，成功后才能返回入队ACK；P可在Fetch前计算并持有KV，但此时不向D发送LOAD或启动Decode Forward。FetchResponse单次接入后复用GenerateStreamCall的LOAD、首token、GENERATE和输出背压逻辑，不缓存整段生成结果。batch_id写入P的TaskInfo。重复ID的整批输入全部拒绝，其他非法项分别返回错误；成功项不因同批其他项失败而回滚。当前每个请求仍按既有单请求Forward调度，尚未模拟多请求合批计算。

入队RPC结束或其客户端取消不终止已接受的请求；generate_config.timeout_ms仍覆盖D分配、P排队、等待Fetch和生成全程。fetch_attach_timeout_ms从入队成功开始计时，非正值沿用原生10分钟默认值，接入后撤销，只约束接入等待。未接入超时、Fetch断流、节点关闭都会取消P/D持有者。Fetch重复接入返回ALREADY_EXISTS，未就绪或过期返回NOT_FOUND。批量入口当前要求文本P/D请求、max_new_tokens > 1；单token、图片、多DP明确不支持。

Cancel只针对原生批量请求，保持真实PrefillBatchRpcServer的作用域；直接GenerateStreamCall仍通过RPC取消。Cancel接受正request_id：活动批量请求返回ACCEPTED并锁定8429/RESOURCE_EXHAUSTED，取消P及当前或稍后建立的D链路；这不代表D已完成资源释放。未登记ID返回TOMBSTONED并阻止随后入队；已完成的请求返回NOT_FOUND。抢占后的Fetch和后续同ID入队仍返回8429。墓碑及完成历史保留10分钟，按到期顺序清理；该期间不支持复用请求ID。Mock仍限制最多1024个活动P/D请求，达到上限明确拒绝；历史项不另加请求数量配额，避免长时间发流后突然拒绝新请求。

动态合批计算、正式Master与Client和性能验收仍属后续工作。上述验证使用Mock原生RPC和真实网络，不等同于与真实引擎的差分验收。

### 共享Decode推进与停止模型

DecodeStepModel只计算一轮提出、接受、提交及验证多少token，不管理RTP协议或KV。默认draft_tokens=0，每轮提交/验证1个token。开启推测时必须显式设置acceptance_rate，随机接受率还必须有seed；不预设某个模型的接受率。按首次拒绝截断连续接受前缀A，再加一个纠正/奖励token，即C=A+1。对长度k、固定独立接受率α，P(A=j)=α^j(1-α)（j<k），P(A=k)=α^k，因此E[C]=1+α+…+α^k。这是[经典推测解码论文§3.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)的封顶几何模型，不是二项分布，也不宣称复现每种MTP/EAGLE或树状验证算法。

Compute按本轮实际验证数收费，输出指标只累计实际提交数。α=0时仍支付验证草稿的开销。KV预留覆盖本轮验证上下文，未提交的草稿不会推进逻辑生成长度或进入新增缓存索引；预留块由同一请求持有到后续使用或终态释放，保持当前真实StreamCacheResource尚不支持部分缩容的边界。接近max_new_tokens时缩短草稿预算。

StoppingModel配置放在Prefill的generation中，默认eos_probability=0，即输出到请求上限；随机概率必须显式seed。它以固定EOS hazard的封顶几何分布近似自然停止，遵守min_new_tokens与max_new_tokens，ignore_eos强制使用上限。固定hazard是Mock假设，不是线上长度的普遍规律；已有真实observed长度或Client明确输出覆盖时，应由Client设置目标并ignore_eos，不再二次随机EOS。

P按稳定request_id为每个请求只确定一次stop_after，通过Mock内部LOAD元数据传给D。随机数不使用节点名、JVM号或共享随机游标；请求身份需稳定。D不能重复配置generation，不重新抽样；接受长度模型只允许放在Decode的decode中。开启推测后跨过停止位置，只提交到stop_after，但仍计实际验证开销。若首token结束，P发最终首帧并在LOAD后发送PREFILL_COMPLETE，D释放且不重复输出首token，不执行Decode Forward。模型按请求身份及轮次确定随机数，容量等待和重新调度不会重新抽样。

### 共享Engine指标

`endpoints.json`的`metrics`给出Prometheus抓取URL，与控制接口共用HTTP端口，仅接受`GET /metrics`。使用Prometheus官方Java simpleclient 0.16.0及TextFormat导出标准0.0.4文本。每个节点有独立注册表，不使用默认全局注册表；没有常驻采样线程，累计量在事件发生时更新，瞬时量在抓取时读取。

EngineMetrics放在共享目录，不依赖RTP协议、缓存实现或调度器。其他Engine使用同一组件，传入事件和当前状态即可。固定标签为`engine`、`role`、`node`，额外维度仅为下表枚举，不把请求ID、cache key或异常文本放进标签。

| 指标 | 类型与额外标签 | 上报时机与含义 |
| --- | --- | --- |
| `mock_engine_requests_started_total` | Counter | 校验并接纳请求后加一；不支持的协议、重复ID和入口超限不计入。 |
| `mock_engine_requests_finished_total` | Counter，`outcome=success/cancelled/timeout/error` | 已接纳请求只记录一次终态；节点关闭计error。这是Engine结果，不能替代Client最终成功率。 |
| `mock_engine_cache_lookup_tokens_total` | Counter | 成功获得KV租约时累计输入token数，包括已命中的部分。 |
| `mock_engine_cache_hit_tokens_total` | Counter | 同一次分配中实际复用的历史前缀token数；后续请求取消不撤销已发生的查找。 |
| `mock_engine_forward_tokens_total` | Counter，`kind=input/compute/output` | 活跃请求的Forward完成时记录输入、实际计算和输出token数。P路径分别为输入长度、输入减命中、1；D的input/compute为本轮验证token数，output为实际提交数。被取消或超时的Forward不补记成功计算。 |
| `mock_engine_forward_seconds` | Histogram | 完成Forward的实际计时，单位秒，排除请求排队时间；提供bucket、sum、count。它不是Client TTFT。 |
| `mock_engine_requests` | Gauge，`state=waiting/running` | 当前排队、执行数量。 |
| `mock_engine_kv_tokens` | Gauge，`state=total/available/free/held` | 当前KV容量，单位token。available含可淘汰的未引用缓存，不能当作物理free。 |

累计量预建为0，抓取不会清零。缓存命中率按hit tokens增量除以lookup tokens增量做token加权，不能平均每请求命中率。Prefill计算TPS可由完成的compute tokens增量除以Forward seconds sum增量得到，带cache TPS使用input tokens；若要墙钟吞吐则取token counter的rate，两者分母不同。共享层只保存观测值，由查询接口选择对应口径。分母为0时没有可判断的比率，不补成100%或0来通过门禁。

这些指标是新框架的公共口径，不冒充KMonitor原指标。语义查询接口负责PromQL，case优先调用接口；Grafana旧大盘的指标映射与Client请求指标尚未接入。独立节点测试的metrics.prom只是一份快照；托管实验的历史数据保存在Prometheus TSDB。

### Prometheus指标查询

托管模式由Runner自动配置查询。若自行提供外部Prometheus，可在RTP组件业务配置中补充两个Mock点分参数，在已有控制端口启用查询：

```yaml
environment:
  prefill:
    engine: rtp
    config:
      # 其余原生配置与Mock容量照常填写
      prometheus.url: http://127.0.0.1:9090
      prometheus.job: experiment
```

外部模式由调用方采集节点endpoints.json中的metrics地址，job应只覆盖本次实验；多个实验共用Prometheus时使用不同job。共享PrometheusQueries绑定EngineMetricsApi的4个读接口和PrometheusApi；EngineApi继承这些读定义，仍是原有公共接口，没有另列一份路由表。只读节点不会宣称支持scale_in。指标名由EngineMetrics提供，查询不另造一套指标名。

| 接口 | 计算口径 |
| --- | --- |
| `engine/cache/hit_rate` | Prefill历史命中token增量／输入token增量，可用engines选择节点；结果0～1。 |
| `engine/prefill/tps` | Prefill实际计算token增量／完成Forward耗时增量。 |
| `engine/prefill/tps_with_cache` | Prefill输入token增量／完成Forward耗时增量，输入包含命中部分。 |
| `engine/decode/tps` | Decode完成Forward的output token增量／墙钟时间。本地D尚未执行请求，不能据此宣称D请求链路已完成。 |
| `prometheus/query` | 原始表达式在固定query_time执行一次，返回scalar或float instant vector，记录标签、数值和sample_time。 |

普通Engine接口自动过滤配置的job和engine，并按接口限定role；先对每条Counter计算增量，再聚合，保留多节点重启Counter reset的正确处理。缓存命中率做token加权，不平均节点比率；空engines集合直接报错，不能退化为选全部。选择器值做PromQL字符串和正则转义。

显式range使用Runner已经记录的绝对start/end，不在网络调用时重新取时钟。后端用单次instant query在end求值，窗口为end-start；例如[-60s,-10s]查询的是过去50秒，终点也在过去。时间精度为Prometheus的毫秒；小于1ms的范围拒绝执行。latest使用检查时刻前60秒内最近两次采样的irate，表示当前采集周期，不平均整分钟；最近一分钟没有足够采样或没有可计算的Forward时会报错。

已产生数据时，case仍这样检查：

```yaml
- check:
    id: prefill_tps
    describe: 检查过去一分钟的Prefill计算速度
    interface: engine/prefill/tps
    args: {range: [-60s, 0s]}
    expect: {gte: 100}
```

prometheus/query不接受外层range，也不调用query_range；若表达式需要观察历史，在PromQL中自行使用范围函数，并返回单点数值。该逃生接口按原始表达式执行，不自动改写job/engine过滤条件。matrix、string、原生histogram返回类型、非有限值、查询警告和后端失败不作为有效门禁数据；普通指标必须恰好返回一个有限值。原始PromQL空向量可返回空samples，但Runner会将门禁记为ERROR。查询不重试，HTTP与Prometheus求值均受当前步骤剩余预算限制。

真实采集集成测试需显式提供本机Prometheus，未提供时只跳过这项集成测试，其余仍执行：

```sh
PROMETHEUS_BIN=/path/to/prometheus ./scripts/check.sh
```

集成测试由Runner启动独立Prometheus和两个RTP JVM，验证正常门禁、门禁FAIL、主动取消和采集器被杀时的数据保留与进程回收。若Prometheus同目录有promtool，正常分支还在服务退出后离线读取TSDB，核对累计值。该测试不替代正式Client或P/D验收。
