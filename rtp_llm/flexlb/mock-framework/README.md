# Mock框架

Python负责case配置编译和执行组织，Java负责组件共享接口与实现。配置编译与本地执行器独立：compile不产生运行副作用；run通过启动配方管理进程，调用Java组件接口并执行门禁。RTP Mock业务适配、多JVM Client播放、指标采集和正式历史查看服务尚未接入，不由测试组件模拟验收。

## 目录与构建

- `python/mock_framework/config/`：YAML读取、配置检查、点分参数展开、步骤编译。
- `java/interfaces/`：组件接口的唯一权威定义、公共类型和清单生成。
- `python/mock_framework/execution/`：本地进程初始化、顺序执行、HTTP调用、门禁比较、收尾和运行记录。
- `java/runtime/`：将接口实现绑定为HTTP端点，按接口清单校验参数与回执，提供单次调用上下文。
- `tests/`：编译与跨语言接口扩展验证；Java测试跟随模块。
- `scripts/`：构建和验证入口。

使用Java21、仓库已有的Maven Wrapper和uv：

```sh
export JAVA_HOME=<Java21目录>
./scripts/check.sh
```

`build.sh`编译Java并运行测试，然后生成`build/interfaces.json`。Python没有另一份组件接口枚举；`runtime/wait`由Python定义为执行器内置动作，不制造Java端点。生成目录不入库；Python依赖锁定在`uv.lock`。Java构建不继承旧Mock或online_eval模块。

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

`expect`支持`eq`、`count`或`gte/lte`上下界。单位为秒的接口允许`2s`这类阈值；集合引用按Java声明类型校验。Master逐实例的Engine集合可与操作回执的Engine集合引用比较，不能通过合并各Master结果掩盖不一致。这里仅编译断言，真正的比较逻辑由执行器实现。

播放配置支持trace时间节奏或uniform总QPS，模式专属字段不混用。输出长度调整支持固定整数及以下分布配置，调用时实际采样、播放和优先级语义归Java Client。当前入口只编译与校验，不生成随机样本或启动请求。

### 输出长度分布

`input.prepare.output_tokens`可为非负整数，也可为带`distribution`的对象。类型、参数和约束在Java `OutputTokens`定义，Python读取生成清单，不维护第二份分布参数表。

```yaml
# 固定值
output_tokens: 128

# 整数均匀分布，包含min/max两个端点
output_tokens: {distribution: uniform, min: 64, max: 256, seed: 42}

# 正态分布：mean/stddev是原始分布参数，单位为token
output_tokens: {distribution: normal, mean: 128, stddev: 32, min: 1, max: 512, seed: 42}

# 对数正态：ln(X)服从N(mu, sigma²)，不是token长度的算术均值/标准差
output_tokens: {distribution: lognormal, mu: 4.5, sigma: 0.8, min: 1, max: 2048, seed: 42}

# 离散选择：weights可省略，省略时等概率
output_tokens: {distribution: choice, values: [64, 128, 512], weights: [1, 2, 1], seed: 42}
```

随机形式必须提供int64的seed。token数与上下界是非负int32，min≤max；正态和对数正态先裁剪到[min,max]，再四舍五入到整数，不进行条件重采样，因此配置的mean不是裁剪后样本均值。stddev/sigma可为0，数值必须有限。离散values非空，weights如提供必须与其等长、有限、非负且至少一个为正，权重不要求加起来等于1。

Java Client实现时应基于seed和稳定逻辑请求身份生成每个请求的一次输出目标，不随JVM数、分片或线程调度顺序改变；同一次请求的重试保留已选目标。覆盖只影响本次输出，不改写数据集settings/observed或输入结构，采样后的实际目标随请求记录保存。此处规定采样语义，实际采样器与多JVM读取在Client批次实现。

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

启动配方是开发接线配置，不放入case业务区域，也不新增一套业务接口定义。当前需要组件提供可启动入口；框架没有默认的RTP或Client业务启动配方。

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

运行目录不覆盖已有目录，保存原始case、编译配置、启动配方、接口清单、代码版本与工作区差异、组件配置和日志、phase/step时间线、门禁实际值及范围、最终结果。PASS表示所执行步骤通过，FAIL表示门禁不满足，ERROR表示执行或数据异常，CANCELLED表示取消；清理异常不能得到PASS。当前尚不产生业务指标或逐请求记录，正式报告的数据保留由后续采集接入完成。测试中的TransportProbe只用于验证真实JVM进程及网络链路，不实现Engine或压测行为。
