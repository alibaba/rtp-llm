# Background
Supports structured pruning [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627). Specifically, for the Attention Layer, the head_num varies layer by layer (where head num can be reduced to 0); for the FFN Layer, the inter_size varies layer by layer.
Applied to [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758).

## Usage

Add two lists in the model's config.json to specify the head_num and inter_size for each layer, and the number of elements in the list needs to be consistent with layer_num.
The config needs to be consistent with the trained ckpt.
For example:
``` json
{
  "layer_head_num": [21, 8, 19, 18, 19, 21, 23, 23, 20, 27, 22, 18, 23, 0, 1, 9, 21, 0, 1, 0],
  "layer_inter_size": [3064, 5652, 9139, 8508, 6197, 5005, 4520, 4586, 4698, 4828, 4776, 4904, 3459, 5282, 7120, 8526, 8475, 9143, 11282, 11428],
  "layer_num": 20
}
```

Then you can call it directly.

Currently, sparse only supports limited models such as Qwen, but it can be easily extended to other models with reference to Qwen.