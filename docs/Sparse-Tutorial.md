# 背景
支持了结构化剪枝 [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)。具体的来说，对于Attention Layer，head_num逐层不同（其中head num可以减到0）；对于FFN Layer，inter_size逐层不同。
应用于[Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758).

## 使用方法

在模型的config.json中增加两个list，列出每一层head_num和inter_size,且list的元素个数需要与layer_num一致。
config需要与训练得到的ckpt保持一致。
例如：
``` json
{
  "layer_head_num": [21, 8, 19, 18, 19, 21, 23, 23, 20, 27, 22, 18, 23, 0, 1, 9, 21, 0, 1, 0],
  "layer_inter_size": [3064, 5652, 9139, 8508, 6197, 5005, 4520, 4586, 4698, 4828, 4776, 4904, 3459, 5282, 7120, 8526, 8475, 9143, 11282, 11428],
  "layer_num": 20
}
```

然后调用即可

当前sparse仅支持了Qwen等有限模型，但参考Qwen可以很轻松的拓展到其他模型。