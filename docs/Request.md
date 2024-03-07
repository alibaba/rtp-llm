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
| `using_hf_sampling` | `bool` | 是否使用hf的采样 |
| `num_beams` | `bool` | beam search的个数 |
| `print_stop_words` | `bool` | 是否打印stop words |
| `stop_words_str` | `List[str]` | stop words的字符串 |
| `stop_words_list` | `List[List[int]]` | stop words的token ids |
| `calculate_loss` | `int` | 是否计算loss,0:不计算，1:单词的loss求和，2:每个token的loss各自返回 |
| `return_hidden_states`/`output_hidden_states` | `bool` | 是否返回hidden_states |
| `return_logits`/`output_logits` | `bool` | 是否返回logits |
| `select_tokens_id` | `List[int]` | 只有在select_tokens_id集合中token 才返回logtis，不指定的话，就返回所有。|
| `select_tokens_str` | `List[str]` | 这个列表中的str，会使用tokenizer encode成对应的token id，append到select_tokens_id中。|
| `return_input_ids` | `bool` | 是否返回input ids |
| `return_output_ids` | `bool` | 是否返回output ids |
| `task_id` | `int` | system prompt的id |
| `chat_id` | `str` | 多轮会话的id |
| `timeout_ms` | `int` | request的超时时间 |
| `request_format` | `str` | request的请求格式，取值(raw/chatapi)|
| `yield_generator` | `bool` | 是否流式输出 |