# 无 GPU 的 ReportEvent mock 集成测试

从仓库根目录执行：

```sh
python3 rtp_llm/cpp/cache/events/test/run_portable_mock_tests.py --sanitize --check-mutations
```

需要 Python 3、Git、支持 C++17 的 Clang（默认）或 GCC，以及访问 GitHub 的网络。
GCC 可以用 `--cxx g++` 指定。无需 GPU、CUDA、Torch、Bazel、模型权重或正在运行的 KVCM。
如果编译器未安装 AddressSanitizer 运行库，可以先去掉 `--sanitize` 执行普通测试。

脚本临时获取 `deps/git.bzl` 固定版本的 GoogleTest 和 RapidJSON，并应用项目已有的 RapidJSON patch。
也可以通过 `--googletest-dir` 和 `--rapidjson-dir` 使用已准备好的同版本源码；提供的目录不会被修改，
其中 RapidJSON 应由调用方事先应用项目 patch。临时依赖、测试程序和故障注入副本都放在项目 `.tmp/`，
正常结束或 Python 捕获异常时自动清理；生产源码不会被修改，也不会启动后台服务或修改引擎配置。

## 真代码与 mock 的边界

直接编译、执行以下生产实现：

- `BlockTree.cc`：插入、替换、删除、初始化扫描、驻留条件判断和事件触发。
- `GroupSetResource.cc`、`KVCacheEventState`：资源层级和每层发布状态。
- `KVCacheEventQueue.cc`：真实入队、出队、容量限制；原有并发测试也一起执行。
- `KVCMEventReport.cc`：事件合并、实际 JSON 报文和跨层全量快照。

测试替换了物理资源池、传输调度、发布线程/HTTP 和远端 KVCM。测试通过真实 `GroupSetResource`
注入成功、失败、取消等传输结果，再让真实 `BlockTree` 决定是否发布。接收端独立解析实际 JSON，
按 `(block_key, medium, spec)` 维护索引，并使用字面量预期检查结果。

`portable/include` 下的 pool/logger/assert doubles 只供此脚本使用，不进入生产 Bazel target。
测试保留缓存树的断言检查，但不模拟物理池的申请、释放或引用计数。

## 新增的 8 个行为场景

1. HOST/DISK 插入和删除产生正确的介质、spec、URI，并到达接收端索引。
2. HBM → HOST → HBM 迁移过程中撤回源，仅在目标完成后发布目标。
3. LOAD_PENDING、LOADING、DEMOTING 回滚恢复 HOST；已 detached 的资源不恢复发布。
4. 成员数量不足或包含空 block 的 DEVICE GroupSet 不上报，补全后恢复 HBM。
5. 同一批次的跨层迁移保留旧层删除与新层新增；真实插入路径替换 HOST 为 HBM 时自动更新两层。
6. 队列溢出或增量丢失后，全量快照修复三层索引，并清理已删除 HOST key。
7. 选择 HBM/HOST 后，增量和快照只包含这两层。
8. 启动时已有 HOST 缓存进入首轮快照；空快照清空旧索引。

同时执行之前的 13 个报文、状态和队列测试，共 21 个测试。
`--check-mutations` 在临时源码副本中分别注入五种错误：漏掉 HOST 发布、发布传输中资源、漏掉插入回调、
合并不同介质的事件、快照遗漏 HOST。每一种都必须触发新增行为测试的断言失败，否则脚本返回失败。

## 不能替代的验证

本测试不执行 `BlockTreeLoader` / `BlockTreeEvictor` 的真实调度回调，因此不能单独证明这些调用点全部接通；
相关原生测试仍在 `BlockTreeCacheTest.cc`。它也不验证 GPU/磁盘数据拷贝、CRC、实际资源生命周期、
完整 `KVCMPublisher` 的 HTTP/重试/并发逻辑，或部署中的 KVCM 服务端版本、快照栅栏与并发清理。

## 2026-11-05 验证结果

21 个测试及其生产依赖在 Mac 上以 `-Wall -Wextra -Werror` 编译成功，普通版和 AddressSanitizer 版均成功构建。
两次执行均在进入 GoogleTest 前被 macOS 安全策略以 `SIGKILL` 终止，系统日志报告签名校验拒绝。

随后通过 ASI 网页终端，在 Ubuntu 24.04 / ARM64、GCC 11.5、Python 3.12.7 的 CPU 环境执行：

```sh
python3 rtp_llm/cpp/cache/events/test/run_portable_mock_tests.py --cxx g++ --sanitize --check-mutations
```

- **21/21 测试通过**，正常实现的 AddressSanitizer 运行无报错。
- **5/5 故障注入均被行为断言捕获**，完整脚本退出码为 0。
- 首次故障注入因 `publish-in-flight` 的匹配文字出现两次而主动中止；收窄到发布条件后完整重跑通过。
- 传输的 27 个文件经 SHA256 校验与本地工作区一致；使用独立临时目录、低优先级编译和 mock 接收方，
  未向现有 KVCM 服务发送事件，远端临时文件已清理。

这些结果覆盖上文声明的 mock 边界；原生 loader/evictor 调度、真实数据传输和线上 KVCM 联调仍未验收。
