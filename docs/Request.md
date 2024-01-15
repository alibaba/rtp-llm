# 请求格式
## 请求样例
```json
{"prompt": "hello, what is your name", "generate_config: {"max_new_tokens": 1000}}
```
我们也支持 OpenAI 样式请求，详见 [OpenAI接口](OpenAI-Tutorial.md)

## 参数列表

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `prompt` / `prompt_batch` (二选一) | `str` / `List[str]`, required | prompt |
| `generate_config` | `dict`, optional, default=`{}` | 生成参数，目前支持如下参数 |
| `using_hf_sampling` | `bool`, optional, default=`False` | 是否使用hf采样 |


## generate_config

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `max_new_tokens` | `int` | 最大生成token数 |
| `top_k` | `int` | top_k采样 |
| `top_p` | `float` | top_p采样 |
| `temperature` | `float` | logits计算参数(温度) |
| `repetition_penalty` | `float` | 重复token惩罚系数 |
| `random_seed` | `long` | 随机种子 |
| `num_beams` | `bool` | beam search的个数 |
| `calculate_loss` | `bool` | 是否计算loss |
| `return_hidden_states`/`output_hidden_states` | `bool` | 是否返回hidden_states |
| `return_logits`/`output_logits` | `bool` | 是否返回logits |
| `yield_generator` | `bool` | 是否流式输出 |
