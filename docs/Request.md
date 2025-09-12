# Request Format
## Request Example
```json
{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}
```
We also support OpenAI-style requests, see [OpenAI Interface](OpenAI-Tutorial.md) for details

## Parameter List

| Parameter Name | Type | Description |
| --- | --- | --- |
| `prompt` / `prompt_batch` (one of two) | `str` / `List[str]`, required | prompt |
| `generate_config` | `dict`, optional, default=`{}` | Generation parameters, currently supports the following parameters |
| `using_hf_sampling` | `bool`, optional, default=`False` | Whether to use hf sampling |


## generate_config

| Parameter Name | Type | Description |
| --- | --- | --- |
| `max_new_tokens` | `int` | Maximum number of generated tokens |
| `top_k` | `int` | top_k sampling |
| `top_p` | `float` | top_p sampling |
| `temperature` | `float` | logits computation parameter (temperature) |
| `repetition_penalty` | `float` | Repetition token penalty coefficient |
| `random_seed` | `long` | Random seed |
| `using_hf_sampling` | `bool` | Whether to use hf sampling |
| `num_beams` | `bool` | Number of beam search |
| `print_stop_words` | `bool` | Whether to print stop words |
| `stop_words_str` | `List[str]` | Stop words strings |
| `stop_words_list` | `List[List[int]]` | Stop words token ids |
| `calculate_loss` | `int` | Whether to calculate loss, 0: no calculation, 1: sum of word loss, 2: return loss of each token separately |
| `return_hidden_states`/`output_hidden_states` | `bool` | Whether to return hidden_states |
| `return_logits`/`output_logits` | `bool` | Whether to return logits |
| `select_tokens_id` | `List[int]` | Only tokens in the select_tokens_id set return logits, if not specified, return all. |
| `select_tokens_str` | `List[str]` | The strings in this list will be encoded by tokenizer into corresponding token ids and appended to select_tokens_id. |
| `return_input_ids` | `bool` | Whether to return input ids |
| `return_output_ids` | `bool` | Whether to return output ids |
| `task_id` | `int` | System prompt id |
| `chat_id` | `str` | Multi-turn conversation id |
| `timeout_ms` | `int` | Request timeout |
| `request_format` | `str` | Request format, values (raw/chatapi) |
| `yield_generator` | `bool` | Whether to stream output |
