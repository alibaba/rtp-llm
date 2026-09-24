# 老版本 Master 接入 Mock 测试框架

版本对照需要保留同一套 Mock、负载和采集方式，只替换 Master。困难通常不在 jar 路径，而在配置：新框架表达的实验条件，必须翻译成目标 Master 真正理解的配置和行为。

以新配置为起点是合理的，但它提供的是**实验意图**，不能直接作为老版本的完整配置模板。兼容配置应以老版本默认值或有来源的部署原文为底，只改写已经确认语义对应的实验条件。新旧版本本来就不同的算法或默认行为，要作为版本差异保留、解释，不能在“适配”时悄悄抹平。

本文面向自带兼容 gRPC 调度服务的老 Master，重点说明 `schemaVersion=1` 与当前配置的衔接。版本边界按实际接口和源码判断，不按发布时间判断；相同 schema 也不保证所有分支具有相同行为。HTTP-only Master 见文末边界。

## 一、三个接入点，两份配置

现有入口已经支持替换，无需修改 case 框架：

| 环境变量 | 作用与限制 | 消费位置 |
|---|---|---|
| `FLEXLB_FT_MASTER_JAR` | Master jar 的绝对路径；在 Python 模块导入时读取，每个对照臂应启动独立 Python 进程 | [`harness.py:52`](../../src/runtime/harness.py#L52)，启动见 `:1373`、`:1433` |
| `FLEXLB_FT_MASTER_CONFIG_FILE` | 读取普通 JSON，整体替换 Master 的 `FLEXLB_CONFIG`，不做字段合并 | [`master_artifact.py:30`](../../src/runtime/master_artifact.py#L30) |
| `FLEXLB_FT_MASTER_SOURCE_COMMIT` | 完整 40 位源码 SHA，记录为 `declared`；未提供时尝试 jar manifest，否则记为未知 | [`master_artifact.py:11`](../../src/runtime/master_artifact.py#L11) |

配套单测在 [`test_master_artifact.py:14`](../../tests/test_master_artifact.py#L14) 和 `:35`，覆盖产物身份、配置覆盖与 schema 1 发现适配。它们使用测试 jar，不能证明某个真实 jar 支持某个字段。

配置覆盖发生在 Master 启动前。Mock 已经从当前 profile、`config_overrides` 等输入生成了自己的 `master_config.json`；它不会读取外部旧配置。路径见 [`harness.py:948`](../../src/runtime/harness.py#L948)、`:1205`、`:1250`、`:1433`。

因此两份配置必须分别审查：

- **Mock 的配置**决定模拟引擎如何运行，继续使用新框架的格式和性能数据；两臂保持一致。
- **Master 的兼容配置**决定目标 jar 如何排队、估时、选机和调度。估时公式等共同条件需要语义一致；若研究的就是估时差异，应把它列为受控变量。

不要用 `EnvSpec.raw_config` 塞入老方言：它也会进入 Mock 配置渲染。跨版本替换使用上表的外部文件入口。也不要把父 shell 的普通环境变量当成 Master 必然继承的配置；Master 环境从 `BASE_MASTER_ENV` 和 `EnvSpec.master_env` 等显式构造，见 `harness.py:942`、`:1234`。

## 二、临时兼容配置如何写

### 先分清共同条件与版本差异

每个待翻译的条件都应回答四件事：新字段在哪里消费、旧字段在哪里消费、两边单位和触发条件是否一致、没有等价落点时怎么办。只在字段名上做替换，无法回答这些问题。

兼容配置承担两层责任：

1. 把本实验明确要求一致或主动改变的条件，落到老版本的等价配置上。包括调度形态、估时模型、亲和阈值等；不支持的组合应停止并重新限定比较范围。
2. 其余配置保留该老版本自己的默认值，或本次要复现的旧部署原文。不能拿新框架 `STRESS_BASE` 的整段值填满旧配置，否则比较中会混入未声明的行为变化。

省略字段也是一种选择：目标 jar 会补自己的默认值，需要把补齐后的值纳入审查。若老版本独有的过滤器恰好是研究对象，保留它属于版本对照；临时关闭它则是另一个明确命名的干预臂。

### 常见代际差异

下面的 schema 1 指具有所列类和字段的实现；接入其他 ref 时，用 `git show "$OLD_SHA":rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/config/<类名>.java` 核对，不能仅凭 schema 数字套表。

当前 schema 为 3，声明见 [`FlexlbConfig:16`](../../../../flexlb-common/src/main/java/org/flexlb/config/FlexlbConfig.java#L16)。它要求的不只是版本号：把新 JSON 的 `schemaVersion` 改成 1，并不会完成下面这些语义转换。

| 配置意图 | schema 1 的落点 | 当前落点与适配原则 |
|---|---|---|
| 分组决策与发送 | `dispatcher.type=BATCH/NON_BATCH`；`BatchDispatcherConfig:10` 同时包含收集窗口和批次限制 | `scheduler.decision` 与 `dispatcher` 已拆成独立轴。不能把 `sb/sn/wb/wn` 四种组合机械映射到旧二选一；单请求、非批发送可先从 `sn` 验证 |
| 排队 | `QueueSchedulerConfig:10` 的 `ordering`、`queueTimeoutMs` | 核对 ordering 和排队超时的消费逻辑后逐项翻译；不要同时复制新的 `decision`、`capacity` 等整组对象 |
| 请求/决策寿命 | `scheduler.lifecycle` 下的 `staleInflightTimeoutMs`、`deliveredNotAcceptedTimeoutMs` 等，见旧 `RequestLifecycleConfig:10` | 当前顶层 `requestLifecycle.request.timeoutMs` 是请求状态静默期限，`decision.lifetime` 是倍率，不是旧字段改名；见当前 [`RequestLifecycleConfig:17`](../../../../flexlb-common/src/main/java/org/flexlb/config/RequestLifecycleConfig.java#L17) |
| Prefill 估时与亲和 | `router.roles.prefill.executionTimeEstimator`、`cacheAffinity.maxExtraTtftMs/minPrefixHitPercent`，见旧 `RoutingConfig:45`、`:166` | 只拷贝双方支持的叶子字段。阈值相同还要求估时单位、候选集合和亲和判断一致；当前声明见 [`RoutingConfig`](../../../../flexlb-common/src/main/java/org/flexlb/config/RoutingConfig.java) |
| Prefill 候选选择与离群过滤 | 旧 `RoutingConfig:105`、`:124` 的 `selector.candidateChoice.outlierRejection`；消费见旧 `CostBasedPrefillStrategy:330` | 当前没有这一配置轴，不能向当前 JSON 塞回旧 key，也不能把旧过滤器顺手关闭。应记录为版本行为差异或独立干预 |

更老的 flat 配置中，`cacheAffinityFirstMaxExtraWorkTokens` 比较的是旧估时评分；嵌套配置的 `maxExtraTtftMs` 表达额外时间预算。即使某个旧公式恰好使数值对应，也不能推广成 tokens 与 ms 可直接照抄。核对旧 `TaskInfo`、`CacheAffinityFirstStrategy` 的实际表达式，再判断是否可换算。

### 旧部署原文从哪里取

复现部署时优先取实际进程的启动输入及其来源，不能把测试模板当成线上原文。Whale Mock bundle 有明确保留点：[`bundle.py:73–81`](../../../whale_mock/bundle.py#L73) 把原始 `FLEXLB_CONFIG` 写入运行目录的 `master-source-config.json`，另行生成供 Mock 使用的 `master-config.json`。应提取前者，同时保存对应 jar 身份、启动环境来源；后者是投影，不能反推原始配置。

这只是可核对的取证路径，不代表任意 bundle 文件都来自生产部署。没有取到目标部署原文时，明确使用“目标源码默认值”，不要用推测补全来源。

## 三、worker discovery 也是兼容接口

当前 `EnvSpec` 默认 `discovery="file"`（[`harness.py:837`](../../src/runtime/harness.py#L837)）。Mock 写出 endpoint 和 discovery JSON，框架把 endpoint 内的 `MODEL_SERVICE_CONFIG` 交给 Master；当前 Master 通过其中的 `discovery_file` 加载域名到 worker 地址的映射。因此这里的 `file` 并不等于“静态地址列表”。见 `harness.py:1205–1228`、`:1257`，以及当前 [`ServiceDiscoveryConfiguration:15`](../../../../flexlb-common/src/main/java/org/flexlb/config/ServiceDiscoveryConfiguration.java#L15)。

部分 schema 1 Master 原生只有 `NoOpServiceDiscovery`：从 `DOMAIN_ADDRESS:<domain>` 读取静态地址，不认识 `MODEL_SERVICE_CONFIG.discovery_file`，也没有 `MOCK_DISCOVERY_FILE` 的消费者。仅替换 jar 和配置，不能自动补齐这项能力。核对旧 `ServiceDiscoveryConfiguration:24`、`NoOpServiceDiscovery:28–53`。

标准动态发现接法使用仓库现成的 [`WhaleFileDiscovery.java`](../../../whale_mock/discovery_adapter/WhaleFileDiscovery.java)：

1. 在独立旧源码 worktree 中编入这个适配类。它实现目标版本的 `ServiceDiscovery`，不改调度算法；已有构建用法见 [`build_legacy_master.py:41`](../../../whale_mock/build_legacy_master.py#L41)。
2. 编排中**显式设 `discovery="discovery_file"`**；scenario YAML 使用 `environment.discovery: discovery_file`。
3. 对 schema 1，框架移除 `MODEL_SERVICE_CONFIG.discovery_file`，设置 `MOCK_DISCOVERY_FILE`。适配类由该属性启用，读取同一个文件；`getHosts()` 重新加载映射，初次无有效文件时报错，后续坏写保留最后有效快照。

第 3 步的框架条件同时要求 schema 1 和显式 `discovery_file`，默认 `file` 不触发。它只是环境变量桥接，**不会修改 jar，也不是任意老版本的自动兼容层**。锚点：[`master_artifact.py:38–42`](../../src/runtime/master_artifact.py#L38)、适配类 `:23`、`:34–65`。

纯静态拓扑也可向 `EnvSpec.master_env` 显式传入旧版需要的 `DOMAIN_ADDRESS:*`；这属于程序化编排，不应假设普通 scenario runner 会透传父 shell 环境，更不能据此测试扩缩容。地址必须取自 Mock 生成的映射，不能猜 `localhost` 或直接把 HTTP/gRPC 端口互换；当前地址生成见 [`MockControlServer`](../../../../flexlb-mock-engine/src/main/java/org/flexlb/mockengine/MockControlServer.java#L519)。

## 四、从源码到两臂启动

### 1. 准备独立构建

先完成[运行底座](build-and-runtime.md)的 Java 21 与 Python 依赖。以下变量使用绝对路径，输出目录应是本次独占的新目录；`OLD_REF` 选择具备上述 schema 1 接口的目标源码。

```bash
export FRAMEWORK=/path/to/current-checkout
export OLD_TREE=/path/to/isolated-old-worktree
export OUT=/path/to/new-output
export JAVA_HOME=/path/to/jdk-21
export PATH="$JAVA_HOME/bin:$PATH"
OLD_REF='<target-ref>'
export OLD_SHA=$(git -C "$FRAMEWORK" rev-parse "$OLD_REF^{commit}")
export NEW_SHA=$(git -C "$FRAMEWORK" rev-parse HEAD)
export OE="$FRAMEWORK/rtp_llm/flexlb/tools/online_eval"
export PYTHONPATH="$OE/src:$OE"
mkdir -p "$OUT"
git -C "$FRAMEWORK" worktree add --detach "$OLD_TREE" "$OLD_SHA"

# 旧版缺少文件发现时，编入已有适配类；接口不同则先停止核对。
ADAPTER="$OE/../whale_mock/discovery_adapter/WhaleFileDiscovery.java"
DEST="$OLD_TREE/rtp_llm/flexlb/flexlb-common/src/main/java/org/flexlb/mockdiscovery"
mkdir -p "$DEST"
cp "$ADAPTER" "$DEST/"
(cd "$OLD_TREE/rtp_llm/flexlb" && ./mvnw -P'opensource,!internal' \
  -pl flexlb-api -am package -DskipTests) >"$OUT/build-old.log" 2>&1
(cd "$FRAMEWORK/rtp_llm/flexlb" && ./mvnw -P'opensource,!internal' \
  -pl flexlb-api,flexlb-mock-engine -am package -DskipTests) >"$OUT/build-new.log" 2>&1
export OLD_JAR="$OLD_TREE/rtp_llm/flexlb/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar"
export NEW_JAR="$FRAMEWORK/rtp_llm/flexlb/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar"
```

检查两个构建的退出状态和 `BUILD SUCCESS` 后再继续。记录两边源码 SHA、工作树修改、适配类 SHA-256 和最终 jar SHA-256；推荐使用干净的固定源码。`SOURCE_COMMIT=declared` 只是声明，不能把加过适配类的 jar 宣称为未经修改的原始制品。框架的身份归档见 `master_artifact.py:45–53`。

### 2. 从新配置翻译一个有限范围的例子

下面只展示 QUEUE/FIFO、单请求非批发送的接入方式，不是通用转换器。共同控制项是队列超时、Prefill 估时和亲和阈值；旧 selector、旧生命周期等保留目标 jar 默认值。用于正式实验前仍需按第二节审查这些版本差异。scenario 带有 overrides 时，应使用同样 overrides 渲染新配置，不能以裸 profile 替代它。

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
from flexlb_cfg import render_env
out = Path(os.environ["OUT"])
new = json.loads(render_env("single-nonbatch"))
assert new["scheduler"]["type"] == "QUEUE"
assert new["scheduler"]["ordering"]["type"] == "FIFO"
assert new["scheduler"]["decision"]["type"] == "SINGLE"
assert new["dispatcher"]["type"] == "NON_BATCH"
p = new["router"]["roles"]["prefill"]
assert p["executionTimeEstimator"]["type"] == "FORMULA"
old = {
    "schemaVersion": 1,
    "scheduler": {"type": "QUEUE", "ordering": {"type": "FIFO"},
                  "queueTimeoutMs": new["scheduler"]["queueTimeoutMs"]},
    "dispatcher": {"type": "NON_BATCH"},
    "router": {"roles": {"prefill": {
        "executionTimeEstimator": {k: p["executionTimeEstimator"][k]
                                   for k in ("type", "expression")},
        "cacheAffinity": {k: p["cacheAffinity"][k]
                          for k in ("maxExtraTtftMs", "minPrefixHitPercent")},
    }}},
}
for name, config in (("new", new), ("old", old)):
    (out / f"{name}.json").write_text(json.dumps(config, indent=2))
PY
```

### 3. 先用目标 jar 校验，再启动

schema 1 的严格解析实现和当前实现都会拒绝未知 key。当前入口见 [`ConfigService:22–64`](../../../../flexlb-common/src/main/java/org/flexlb/config/ConfigService.java#L22)，schema 校验见 [`FlexlbConfigValidator:98`](../../../../flexlb-common/src/main/java/org/flexlb/config/FlexlbConfigValidator.java#L98)。不要假定所有老 jar 都忽略未知字段，也不要用当前解析器替老 jar 背书。

对具有 `ConfigService.parse(String)` 的 Spring Boot jar，可直接执行实际制品中的解析器。以下命令对 old/new 各跑一次，`ARM=old` 时 `JAR=$OLD_JAR`，`ARM=new` 时 `JAR=$NEW_JAR`：

```bash
export ARM=old JAR="$OLD_JAR"
python3 - <<'PY'
import os, zipfile
from pathlib import Path
dest = Path(os.environ["OUT"]) / ("unpack-" + os.environ["ARM"])
with zipfile.ZipFile(os.environ["JAR"]) as z:
    for name in z.namelist():
        if name.startswith(("BOOT-INF/classes/", "BOOT-INF/lib/")):
            z.extract(name, dest)
PY
cat >"$OUT/ParseConfig.java" <<'JAVA'
import java.nio.file.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.ConfigService;
class ParseConfig {
    public static void main(String[] args) throws Exception {
        var config = ConfigService.parse(Files.readString(Path.of(args[0])));
        new ObjectMapper().writerWithDefaultPrettyPrinter()
            .writeValue(Path.of(args[1]).toFile(), config);
    }
}
JAVA
java --class-path "$OUT/unpack-$ARM/BOOT-INF/classes:$OUT/unpack-$ARM/BOOT-INF/lib/*" \
  "$OUT/ParseConfig.java" "$OUT/$ARM.json" "$OUT/$ARM-parsed.json"
```

退出成功才表示目标 parser 接受配置。再用副本加入一个不存在的顶层 key、一个不存在的 Prefill key，分别确认拒绝；若目标版本行为不同，按实际结果记录并加强字段消费检查。`*-parsed.json` 用于观察补齐默认值后的对象，不是可再次投喂的配置模板，序列化可能带有派生字段或 null。

### 4. 同一个编排，分别启动两种 Master

保存下列程序为 `$OUT/smoke.py`。它验证发现、存活和一条完整请求，不评价性能优劣。接口见 [`EnvManager`](../../src/runtime/harness.py#L1033) 与 [`EngineOps.run_one_request`](../../src/runtime/engine_ops.py#L778)。

```python
import json, sys
from pathlib import Path
from runtime.harness import EnvManager, EnvSpec, http_post_json
from runtime.engine_ops import EngineOps

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
manager, ops = EnvManager(root), None
try:
    env = manager.ensure(EnvSpec(
        label="compat", run_dir=root / "environment",
        n_prefill=1, n_decode=1, discovery="discovery_file",
        master_profile="single-nonbatch", mock_heap="512m",
        event_loop_threads=2, completion_threads=2,
        master_jvm_args=["-Xms128m", "-Xmx2g"],
        master_extra_args=["--flexlb.log.path=" + str(root / "logs")],
    ))
    status, info = http_post_json(env.master_http("/rtp_llm/master/info"), {})
    (root / "master-info.json").write_text(json.dumps(info, indent=2))
    assert status == 200 and info["ready"], info
    ops = EngineOps("127.0.0.1", env.master_http_port,
                    env.mock_http_port, env.master_management_port)
    address, error = ops.run_one_request(90001, input_len=2048, output_len=2)
    (root / "request.json").write_text(json.dumps({"prefill": address, "error": error}))
    assert error is None, error
finally:
    if ops is not None:
        ops.close()
    manager.teardown()
```

```bash
FLEXLB_FT_MASTER_JAR="$OLD_JAR" FLEXLB_FT_MASTER_SOURCE_COMMIT="$OLD_SHA" \
FLEXLB_FT_MASTER_CONFIG_FILE="$OUT/old.json" python3 "$OUT/smoke.py" "$OUT/A"
FLEXLB_FT_MASTER_JAR="$NEW_JAR" FLEXLB_FT_MASTER_SOURCE_COMMIT="$NEW_SHA" \
FLEXLB_FT_MASTER_CONFIG_FILE="$OUT/new.json" python3 "$OUT/smoke.py" "$OUT/B"
```

检查两臂 `master-info.json` 中 P/D 的 discovered、alive 数量与拓扑一致，以及 `request.json` 无错误。端口打开或 `/master/info` 能返回不等于完成了请求链路验证。

接入现有 scenario 时，将同样三个变量加在[场景 runner](scenario.md)命令前，两臂选相同实例、profile、性能预设、流量输入和资源预算，使用独立输出目录与 `--parallel 1`；临时 YAML 的 `environment.discovery` 显式设为 `discovery_file`。每个臂单独启动进程，不能在已经 import harness 的进程里换环境变量来切 jar。初次接入先验证协议，再开展长时间 A/B。

## 五、配置自检与“写了但没用”的防御

`environment/actual-master-config.json` 是**传给 Master 的启动原文**，`master-artifact.json` 中的 `effective_config_sha256` 对该原文取 hash。它们证明启动输入，不是 Master 解析后的权威回读。代码在 [`master_artifact.py:45–53`](../../src/runtime/master_artifact.py#L45)。

两臂启动后必须完成以下核对：

1. 检查 jar SHA、源码声明和配置 hash；确认实际启动原文与各自临时文件一致。
2. 对两份 actual config 做逐字段 diff，再对目标 jar 解析出的默认值做审查。可先用 `python3 -m json.tool --sort-keys` 格式化后 `diff -u`，不要只比较原始文件的文本行序。
3. 给每项差异归因：**方言/结构转换、已声明的受控变量、明确保留的版本行为或默认值差异**。跨 schema 的原始 JSON 不可能只有受控变量不同；真正的判据是没有无法解释的语义差异。排队、超时、估时、候选过滤等重要项目应逐项写进本次运行的对照记录。
4. 对关键旋钮做阳性对照：亲和预算应在有缓存候选与额外等待的负载下改变选择；过滤阈值应在制造对应 pending/wait 差异后改变候选或拒绝原因；超时应在刻意延迟后改变终止时刻；发现应在成员变化后改变发现/存活数。只验证解析值变化不够，负载必须能触发相应路径。

若要确认进程内完整对象，权威来源是目标版本 `ConfigService.loadBalanceConfig()` 的实际返回值（当前 `ConfigService:66`），可在受控调试环境只读检查。当前没有可直接替代它的完整配置 HTTP 回读接口；启动日志只摘要部分配置，[`/rtp_llm/debug/snapshot`](../../../../flexlb-api/src/main/java/org/flexlb/httpserver/DebugSnapshotService.java#L38) 展示运行状态，不能当配置全量导出。解析成功也不证明调用链使用了字段，应继续查 getter 的消费者和行为响应。

## 六、范围与排障

| 现象 | 优先检查 |
|---|---|
| unknown property / schema 不匹配 | 是否把新配置原样交给旧 jar；是否由目标 jar 解析；是否复制了整组新字段 |
| Master 启动但没有 worker | 老 jar 是否包含发现实现；是否显式选择 `discovery_file`；是否生成且传入了同一个映射文件 |
| worker 有地址但请求失败 | gRPC 服务、方法和消息是否兼容；地址/端口是否来自 Mock 产物；查看两端异常，不绕过 readiness |
| 配置值变了，结果没变 | 实际启动文件、默认值、getter 消费路径和触发负载；不要先断言“未知 key 被忽略” |
| A/B 不可比 | Mock 配置是否一致；是否用新默认覆盖了旧行为；是否遗漏了兼容配置绕过的 scenario overrides |

HTTP-only 或调度协议不兼容的更老 Master 不在本接法内。仅翻译配置不能解决传输协议差异；实验目录中的 `build-mock-http`、`fix_client_http_schedule.py`、`fix_http*` 等临时配方需要单独维护和验证，不能当作当前框架正式支持的能力。具体实验出处与验证产物保留在运行归档，不将其补丁或运行参数复制进本通用文档。
