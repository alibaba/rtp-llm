# SCR 运行状态修正与同类问题排查

本补丁基于开源提交 `1e3627051f17eedb09c49b14037b82e277052c99`。修复范围是进程运行身份及其已知消费者；本地验证通过不代表完整镜像或真实跨机 restore 已通过。

## 问题与执行顺序

CRIU 恢复的内存包含 seed 的环境变量、单例、配置副本、连接和文件描述符。构造函数通常不会重新执行。仅更新环境变量，也不会自动修正已经派生出来的成员字段。

原 C++ Logger 构造时读取 `GetDefaultIp(ip_)`，随后一直用这个字段生成前缀。最小复现保留 Logger 对象，把网络地址由 `192.0.2.10` 改成 `.20`，下一条日志仍然输出 `.10`。

统一入口位于 [scr_runtime_fixup.py](../rtp_llm/utils/scr_runtime_fixup.py)：

```text
Epsilon 屏障成功返回
  → fixup_runtime_after_restore(generation, lifecycle)
      → 每次读取恢复输入，校验支持的字段
      → 获取本轮 Pod IP，更新进程环境
      → 刷新 C++ Logger 和已加载的 HippoHelper 身份缓存
      → lifecycle.restore_fixup：RPC 地址、前端身份等组件修正
  → lifecycle.release_template：启动指标等延迟组件
  → 启动服务监听和 NormalEngine 循环
```

入口也覆盖 seed 在 checkpoint 后继续运行的情况；不依赖可能被 dump 保存的 SCR_PHASE 判断是否应跳过修正。普通非模板路径保持原逻辑。每次恢复都重新读取输入，即使多次恢复使用同一个 checkpoint generation，也不跳过。记录的上轮身份仅供消费者和审计使用，不作为下轮发现地址的输入。

输入、原生接口或组件 fixup 失败都会抛错，正常 release 不会执行；原有 abort 清理逻辑仍会运行。这不是所有组件状态都能回滚的事务。修正期间失败的进程不能被作为可服务的成功 restore 验收。

## Logger 的修复

[Logger.cc](../rtp_llm/cpp/utils/Logger.cc) 新增 `Logger::refreshRuntimeIdentity(ip)`，通过 [原生绑定](../rtp_llm/cpp/pybind/init.cc) 暴露 `refresh_logger_after_scr`。

- 每个进程使用一个不可变 IP 快照，通过原子 shared_ptr 读写发布，避免与并发日志线程直接读写同一个 string。
- engine、query access、stack trace 等已有 Logger，以及恢复后延迟创建的 Logger 都读取同一个快照；普通和 trace 前缀均覆盖。
- 不重建 appender、不覆盖旧日志、不改变 rank；不带前缀的原始 access 日志格式保持原样。
- CPU/frontend 进程不会为了日志主动加载 CUDA 扩展；已加载但缺少新接口的旧 native 库会明确失败。因此交付必须使用匹配的 Python 与 native 构建，不能只热补 Python。

## 后续 restore env 文件的接入点

`register_restore_env_provider(provider)` 注册进程级回调。回调接收 `generation`，在每次 fixup 时读取平台提供的文件并返回 `Mapping[str, str | None]`。也可以在直接调用统一函数时传入 `restore_env` 映射，用于测试或明确的集成入口。

目前没有定义文件路径、格式，也没有假设该文件已经存在。接入时必须：

1. 在每个参与进程的屏障之前注册 reader，文件内容在恢复后读取，不在 seed 中预读缓存。
2. reader 校验恢复输入与当前 Pod / 本次恢复的绑定关系，拒绝过期文件；平台原子发布完整文件。相同 checkpoint generation 可以被重复恢复，不能仅凭 generation 相同认定文件新鲜。
3. 从完整环境中选择已支持字段，作为数据解析，不能 shell `source`。
4. 当前支持 RequestedIP、HIPPO_SLAVE_IP、HIPPO_ROLE、HIPPO_ROLE_SHORT_NAME、HIPPO_APP、HIPPO_SERVICE_NAME、kmonitorSinkAddress、kmonitorPort。None 删除对应 seed 环境值；例如删除旧短 role，允许新的 HIPPO_ROLE 生效。新增字段必须同时检查它的派生缓存和使用者。
5. 明确提供的 RequestedIP 作为当前 Pod 地址；未提供或显式删除时重新解析当前网络身份，绝不回退到旧 RequestedIP。当前 Python 外部地址契约为非 loopback IPv4。

未提供 HIPPO_SLAVE_IP 时无法从 Pod IP 推导宿主机 IP；代码保留现有环境并记录其新鲜度未验证。该问题需要平台文件输入后才能闭环。GPU 拓扑、模型参数、凭证、路径和 SCR 控制变量不在这个输入接口的覆盖范围。

## 同类状态排查

以下结论来自此基线代码检查，未使用新的线上运行结果替代证明。

| 状态 | 当前机制与本次处理 | 仍需关注 |
| --- | --- | --- |
| C++ Logger.ip_ | 本次改为全进程共享、显式刷新的身份快照 | 完整 native 镜像和实际日志验收待做 |
| Python HippoHelper 类级 host/container/role/app/group | 已有刷新方法，本次在统一入口先刷新；后续刷新复用本轮 Pod 身份 | host 等环境输入仍需平台提供；app_workdir 属于路径语义，未开放修改 |
| native Kmonitor 的 RequestedIP、sink、标签 | 已有暂停/重建/发布时刷新机制；本次确保先应用恢复环境，并避免重新解析覆盖明确提供的 Pod IP | HIPPO_SLAVE_IP、kmonitorSinkAddress 不刷新时仍可能连旧宿主机 |
| BackendRPCServerVisitor.source_ip | 构造时复制 server_config.ip；本次把身份刷新从 local-comm 扩展到所有走此模板入口的拓扑 | 地址成员仍由 endpoint manifest 决定；不把请求来源 IP 当作 rank 地址 |
| Cache-store 对外地址与本机 loopback 通信 | 本次让 local-comm 的对外广告复用同一轮 Pod 身份；显式 manifest 地址保留 | 验收继续要求实际跨 Pod KV 传输 |
| RemoteRpcServer.process_id_ / peer / cache-store | 现有代码延迟到 release 才读取 IP/PID、初始化 peer 和 transport | 不等于任意已运行通信对象都支持重建 |
| ServerConfig.ip、DistributedServer.worker_info / TCPStore / NCCL 配置 | 多处启动时缓存；现有模板路径通过 endpoint manifest 和限定的 loopback 模式处理主要服务地址 | 不能把 manifest 更新等同于现存 TCPStore/NCCL 通信器都已重建；多节点和运行中 checkpoint 需要单独验证 |
| HostService / VipServerWrapper.hosts / MasterService 路由快照与线程 | 构造阶段会发现地址并缓存，MasterService 创建刷新线程；本轮未看到这些类自身的完整模板 prepare/fixup/release 实现 | 明确列为后续审计项；周期刷新不能代替恢复前暂停、恢复后刷新、就绪门禁 |
| MasterClient._channels | 按 target 缓存异步 gRPC channel，惰性创建 | seed 未接流量可减少捕获连接；运行中 checkpoint 必须处理已有 channel 的关闭/重建 |
| GrammarValidator 沙箱池 | 现有 prepare 清理子进程与连接、release 恢复池目标 | 属于资源生命周期修复，不能用 env/string 刷新替代 |

## 验证与交付界限

新增两组可在 CPU 主机运行的测试：

```bash
python -m unittest rtp_llm.utils.test.scr_native_logger_test rtp_llm.utils.test.scr_runtime_fixup_test -v
```

native 测试编译真实 Logger.cc/Logger.h，alog/autil 使用明确的测试替身；检查旧实例、延迟创建实例、连续两次身份变化、trace 前缀、非法输入和并发读写。它不验证真实 alog 文件句柄、pybind 动态加载或 CRIU。

Python 测试通过真实模板入口验证顺序、重复恢复、环境字段校验与删除、旧 native 拒绝、失败阻止正常 release，以及指标 / KV / 前端身份的一致性。原有 Kmonitor、local-comm 和生命周期测试纳入本地回归。

本次核心回归 36 项通过，Python 语法、文档本地链接和 `git diff --check` 通过。测试使用独立的 `../fixup-test-venv`，补充了仓库锁定的 thrift 0.20.0，没有修改全局依赖。

原有 `scr_template_utils_test.test_before_callback_uses_captured_device` 因 `_FakeEpsilon` 缺少 `is_available` 失败，已在未修改的基线独立复现；未修改该测试。涉及真实配置对象的 endpoint 测试依赖本机缺少的 libth_transformer，不能作为本地已通过项目。

完整交付还需：构建匹配 Python/native 的镜像，重新生成 seed，做跨机、同机多次及 worker-kill 后恢复。用同一 attempt 的 UID/CID、restore 边界之后新写日志、Request ID 和真实 peer IP 关联证据。严格要求新日志使用新 Pod IP，检查所有 rank；同时检查实际 Kmonitor sink/tag 和业务输出，不能再仅靠 seed IP 别名选择日志宣称 Logger 已修复。
