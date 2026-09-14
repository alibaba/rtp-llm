这是一个明确的 bug，分类为：FlashInfer 集成层的临时 workspace 容量估算错误。
它不是 MiMo 注意力计算公式错误，也不是并发请求直接造成的，而是 RTP-LLM 将 FlashInfer 推荐的 128 MiB 当成了所有模型都适用的固定容量。
修改前逻辑
原先 [py_flashinfer_mha.py (line 35)](/data1/renkun.ren/RTP-LLM/github-opensource-mimo_v25/rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py:35) 的逻辑等价于：
DEFAULT_WORKSPACE_SIZE_MB = 128

def get_workspace():
    if pool:
        return pool.pop()
    return torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
存在两个问题：
1. 容量固定为 128 MiB
   没有考虑：
   - 本地 Q 头数量
   - 本地 KV 头数量
   - GQA 分组比例
   - V head dimension
   - GPU SM 数量
   - FlashInfer split-KV 调度产生的中间结果数量
2. workspace 池不检查容量
   原来只要池中存在 buffer 就直接 pop()。即便后续加入了动态计算，也可能为需要大空间的 attention 组取回一个旧的 128 MiB buffer。
FlashInfer 实际需求
FlashInfer 在 paged prefill 的 plan() 阶段决定是否进行 split-KV。启用 split-KV 后，需要两个 FP32 临时张量：
tmp_v:
Q heads × expanded batch × Q tile × V head dim × 4

tmp_s:
Q heads × expanded batch × Q tile × 4
本次失败日志明确显示：
batch_prefill_tmp_v size = 139460608 bytes
workspace available      = 134217728 bytes
也就是：
tmp_v       = 133.00 MiB
tmp_s       ≈   1.04 MiB
总需求      ≈ 134.04 MiB
原始容量    = 128.00 MiB
仅 tmp_v 就已经超过整个 workspace，因此必然失败。错误位置见 [GSM8K server log (line 9847)](/home/renkun.ren/log/mimov25/gsm8k_server_20260911_150619.log:9847)。
为什么 MiMo 容易触发
MiMo V2.5 的 GA 层在 TP=4 后大致是：
本地 Q heads  = 16
本地 KV heads = 1
GQA ratio      = 16
V head dim     = 128
这个 GQA 比例比较大。FlashInfer 拆分 KV 后，每个 Q head 都要保存更多分块计算的中间结果，因此 tmp_v 随下面这些量增长：
Q heads × (2 × SM数量 / KV heads) × Q tile × V dim
较短请求可能不会采用相同的 split-KV 计划，或者临时空间较小，所以前 43 个请求可以成功。遇到较长请求后，计划需要约 134 MiB，固定的 128 MiB 才暴露问题。
故障传播
workspace 溢出之后的链路是：
FlashInfer plan 失败
    ↓
PyFlashinferPagedPrefillImpl 构造失败
    ↓
其他候选 attention 实现也不支持当前 MiMo 参数
    ↓
can not find mha type
    ↓
TP worker 异常退出
    ↓
后续请求出现 HTTP 500 / 503 / connection refused
因此 GSM8K 最后的 157 failed_requests 和 21% 得分不能用于评价模型精度，服务在中途已经崩溃。
Bug 评估
这是 RTP-LLM 侧的 bug，理由如下：
- 对合法模型配置和合法请求会确定性崩溃。
- 错误由模型形状和硬件规模决定，不是偶发显存不足。
- FlashInfer 已经正确算出了实际需求，RTP-LLM 没有据此提供足够空间。
- 128 MiB 是常用或推荐初始容量，不是 FlashInfer 保证的最大需求。
- 通过降低并发或手动扩大常量只能规避部分输入，不能从根本上解决。
严重程度可以定为：
类型：资源容量估算错误
严重性：高
影响：模型服务进程退出，后续全部请求失败
数据破坏：无
精度影响：服务崩溃后产生无效评测结果
当前修复
现在改成按 FlashInfer 非 CUDA Graph prefill 调度上界自动计算：
expanded_batch = (2 × GPU SM数量) / 本地KV头数

workspace =
    本地Q头数
  × expanded_batch
  × 最大Q tile
  × (V head dim + 1个LSE值)
  × sizeof(float)
然后与原来的 128 MiB 最小值取较大值，并按 MiB 向上对齐。
以 H20 的 78 个 SM 计算，MiMo GA 会自动得到：
估算容量 = 158 MiB
实际需求 = 134.04 MiB
workspace 池也改成按照“设备一致并且容量足够”选择 buffer。回归用例覆盖了本次 MiMo TP4 的具体参数和日志需求。
这次修复针对当前日志中 enable_cuda_graph=0 的实际执行路径；没有通过环境变量或模型专用常量手动配置容量。