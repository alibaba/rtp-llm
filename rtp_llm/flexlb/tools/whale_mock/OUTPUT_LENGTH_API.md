# 输出长度热更新

Mock 控制端口提供 `GET /output_length` 和 `POST /output_length`。
这是 EOS 停止长度模型，不修改发给 master 的请求长度或调度配置。

```json
{
  "eos": {
    "enabled": true,
    "distribution": "geometric",
    "mean_tokens": 770,
    "seed": 20260913
  }
}
```

POST 完整替换 EOS 配置，校验通过后才应用。均值必须为有限数值且至少为1，seed为64位整数；不支持的分布或未知字段返回400。默认更新当前控制进程的所有P/D引擎；可加 `"engine":"prefill-0"` 定向调试。

P也需要更新：本地PD链路在P生成请求shape时抽样输出目标，并把目标传给D。只更新D不改变P已确定的长度。独立Pod模式需分别调用对应控制接口，接口不负责跨Pod事务。

GET 返回 `engines.<name>.eos` 和 `runtime_override`。更新前保存GET中的EOS对象，回退时作为POST的 `eos` 值重新发送；`{"eos":{"enabled":false}}` 关闭模型、恢复原有请求长度行为。

- 只改变之后新生成的请求shape；已生成shape、排队和在途请求的输出目标不变。
- 保留 `min_new_tokens`、`max_new_tokens`、`ignore_eos` 和显式回放长度规则。
- `mean_tokens`是几何分布参数，最小长度、上限和请求策略会改变实际观测均值；不是保证每个请求输出该长度。
- 缓存、P/D性能公式、scale和集群规模不变，无需重启。
- 更新在每个模型内通过不可变对象一次替换；多个引擎逐一应用，不保证跨引擎同时切换。切换期间重新创建的请求可能使用新参数。
- 热更新仅在当前进程有效，不写环境变量或仓库。重启后恢复启动配置；动态新增引擎使用其启动模型，扩容后应查询并重新下发。
