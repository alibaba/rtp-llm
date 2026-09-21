# 框架结构

```text
config/scenarios/*.yaml       用例数据、拓扑、阈值和时间预算
        ↓
src/flexlb_test_framework/case_programs/*.py     步骤、分支和断言
        ↓
scenario compiler      类型检查并生成执行计划
        ↓
parallel runner        分配 lane，启动 Master/Mock，执行并清理
        ↓
result/report          原始证据、分析 JSON 和 HTML
```

压测使用同一 Java Master、Mock Engine、流量和报告组件，但由 `scripts/stress/run_online_eval.sh` 组织固定流程。功能与场景共享 `parallel_runner.py`：`core` 和 `functional` 都选择 5 个核心合同，`workload` 增加持续流量、阶段观测和 Prometheus 证据。旧扩展功能矩阵已删除，历史版本由 Git 保存。

`config/mode_profiles.yaml` 只定义运行时形态和观测默认值；`config/suites.yaml` 定义 core 选择、functional/workload 分类、采集档位和覆盖关系；场景 YAML 不允许嵌入任意 Python 流程。

资源句柄包含环境代次。环境重建后，旧句柄只能作为历史证据读取，不能继续操作新进程。Schedule、流消费、业务 FINISHED 和资源释放是不同事实，测试必须分别取证。

扩展方式见[新增 case](adding-cases.md)。历史设计和阶段验证不随源码发布，需要时从 Git 历史读取。
